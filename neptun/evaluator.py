"""Evaluation of a trained DNBN system."""

import torch


def _update_confusion(confusion, pred, target, num_classes):
    idx = (target * num_classes + pred).to(torch.int64)
    bins = torch.bincount(idx, minlength=num_classes * num_classes)
    confusion += bins.view(num_classes, num_classes).cpu()


def _macro_precision_f1_from_confusion(confusion):
    confusion = confusion.to(torch.float32)
    tp = torch.diag(confusion)
    fp = confusion.sum(dim=0) - tp
    fn = confusion.sum(dim=1) - tp
    eps = 1e-12

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)

    return precision.mean().item(), f1.mean().item()


def evaluate_system(system, test_loader, device='cpu'):
    """Evaluate the system on a test set.

    Returns:
        (results, comm_stats)
    """
    system.to(device)
    system.eval()
    criterion = torch.nn.CrossEntropyLoss()

    node_ids = list(system.nodes.keys())
    stats = {
        nid: {'correct': 0, 'total': 0, 'loss': 0.0, 'confusion': None}
        for nid in node_ids
    }
    ens = {'correct': 0, 'total': 0, 'loss': 0.0}
    ens_confusion = None

    system.comm.reset_stats()

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            outputs = system(data)

            for nid, output in outputs.items():
                loss = criterion(output, target)
                pred = output.argmax(dim=1)
                stats[nid]['loss'] += loss.item()
                stats[nid]['correct'] += pred.eq(target).sum().item()
                stats[nid]['total'] += target.size(0)
                if stats[nid]['confusion'] is None:
                    num_classes = output.shape[1]
                    stats[nid]['confusion'] = torch.zeros(
                        num_classes, num_classes, dtype=torch.int64
                    )
                _update_confusion(stats[nid]['confusion'], pred, target, output.shape[1])

            avg_logits = sum(outputs.values()) / len(outputs)
            ens_loss = criterion(avg_logits, target)
            ens_pred = avg_logits.argmax(dim=1)
            ens['loss'] += ens_loss.item()
            ens['correct'] += ens_pred.eq(target).sum().item()
            ens['total'] += target.size(0)
            if ens_confusion is None:
                ens_confusion = torch.zeros(
                    avg_logits.shape[1], avg_logits.shape[1], dtype=torch.int64
                )
            _update_confusion(ens_confusion, ens_pred, target, avg_logits.shape[1])

    comm_stats = system.comm.get_comm_stats()

    num_batches = len(test_loader)
    results = {}
    for nid, s in stats.items():
        precision_macro, f1_macro = _macro_precision_f1_from_confusion(s['confusion'])
        results[nid] = {
            'accuracy': s['correct'] / s['total'],
            'loss': s['loss'] / num_batches,
            'correct': s['correct'],
            'total': s['total'],
            'precision_macro': precision_macro,
            'f1_macro': f1_macro,
        }

    ens_precision_macro, ens_f1_macro = _macro_precision_f1_from_confusion(ens_confusion)
    results['ensemble'] = {
        'accuracy': ens['correct'] / ens['total'],
        'loss': ens['loss'] / num_batches,
        'correct': ens['correct'],
        'total': ens['total'],
        'precision_macro': ens_precision_macro,
        'f1_macro': ens_f1_macro,
    }

    return results, comm_stats
