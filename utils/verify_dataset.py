from glob import glob

def verify_dataset():
    """
    Checks if the dataset is valid:
    1. Verify that the number of items in each subset is as expected.
    2. Ensure that labels and original images in each subset match each other.
    """
    
    # Paths for training, validation, and test sets
    train_lab = glob("./dataset/train/train-label-img/*")
    train_org = glob("./dataset/train/train-org-img/*")
    val_lab = glob("./dataset/val/val-label-img/*")
    val_org = glob("./dataset/val/val-org-img/*")
    test_lab = glob("./dataset/test/test-label-img/*")
    test_org = glob("./dataset/test/test-org-img/*")

    # Verify the number of items in each set
    assert len(train_lab) == len(train_org) == 1445, "Training set sizes do not match."
    assert len(val_lab) == len(val_org) == 450, "Validation set sizes do not match."
    assert len(test_lab) == len(test_org) == 448, "Test set sizes do not match."

    # Verify that labels and original images match
    assert set([t.split('/')[-1].split('_')[0] for t in train_lab]) == \
           set([t.split('/')[-1].split('.')[0] for t in train_org]), "Training labels and images do not match."
    assert set([t.split('/')[-1].split('_')[0] for t in val_lab]) == \
           set([t.split('/')[-1].split('.')[0] for t in val_org]), "Validation labels and images do not match."
    assert set([t.split('/')[-1].split('_')[0] for t in test_lab]) == \
           set([t.split('/')[-1].split('.')[0] for t in test_org]), "Test labels and images do not match."

    print("Dataset verification passed!")


if __name__ == "__main__":
    verify_dataset()
