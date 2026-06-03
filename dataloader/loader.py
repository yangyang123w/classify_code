from torch.utils.data import DataLoader

from dataloader.dataset import CsvImageClassificationDataset, read_split_csv


def get_dataloaders(args):
    _, class_names = read_split_csv(args.csv_path)
    train_set = CsvImageClassificationDataset(args.csv_path, args.train_split, class_names, augment=True)
    val_set = CsvImageClassificationDataset(args.csv_path, args.val_split, class_names, augment=False)
    test_set = CsvImageClassificationDataset(args.csv_path, args.test_split, class_names, augment=False)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader, test_loader, class_names
