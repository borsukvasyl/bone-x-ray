from bone_xray.data import parse_mura_dataset


def test_parse_mura_dataset():
    dataset_labels_path = 'tests/data/dataset_labels.csv'
    prefix = 'tests/data'
    result = parse_mura_dataset(dataset_labels_path, prefix)
    expected = [{'images':
                 ['tests/data/MURA-v1.1/valid/XR_WRIST/patient11185/study1_positive/image1.png'],
                 'study': 'MURA-v1.1/valid/XR_WRIST/patient11185/study1_positive/', 'label': 1, 'part': 'XR_WRIST'},
                {'images':
                 ['tests/data/MURA-v1.1/valid/XR_WRIST/patient11186/study1_positive/image1.png'],
                 'study': 'MURA-v1.1/valid/XR_WRIST/patient11186/study1_positive/', 'label': 1, 'part': 'XR_WRIST'},
                {'images': ['tests/data/MURA-v1.1/valid/XR_WRIST/patient11186/study2_positive/image1.png'],
                 'study': 'MURA-v1.1/valid/XR_WRIST/patient11186/study2_positive/', 'label': 1, 'part': 'XR_WRIST'}]
    assert result == expected
