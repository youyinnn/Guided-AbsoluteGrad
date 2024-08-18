import os
import argparse
import shutil

"""
Source: https://github.com/LUSSeg/ImageNet-S
"""


def make(mode, imagenet_dir, save_dir):
    current_dir = os.path.dirname(__file__)

    assert mode in ['50', '300', '919']
    validation_save_dir = os.path.join(
        save_dir, 'validation')
    # test_save_dir = os.path.join(save_dir, 'test')

    if not os.path.exists(validation_save_dir):
        os.makedirs(validation_save_dir)
    # if not os.path.exists(test_save_dir):
    #     os.makedirs(test_save_dir)

    categories = os.path.join(
        current_dir, 'categories', 'ImageNetS_categories_im{0}.txt'.format(mode))
    validation_mapping = os.path.join(
        current_dir, 'mapping', 'ImageNetS_im{0}_mapping_validation.txt'.format(mode))
    test_mapping = os.path.join(
        current_dir, 'mapping', 'ImageNetS_im{0}_mapping_test.txt'.format(mode))
    with open(categories, 'r') as f:
        categories = f.read().splitlines()
    with open(validation_mapping, 'r') as f:
        validation_mapping = f.read().splitlines()
    with open(test_mapping, 'r') as f:
        test_mapping = f.read().splitlines()

    for cate in categories:
        os.makedirs(os.path.join(validation_save_dir, cate), exist_ok=True)
        # os.makedirs(os.path.join(test_save_dir, cate), exist_ok=True)

    for item in validation_mapping:
        src, dst = item.split(' ')
        shutil.copy(os.path.join(imagenet_dir, src),
                    os.path.join(validation_save_dir, dst))

    # for item in test_mapping:
    #     src, dst = item.split(' ')
    #     shutil.copy(os.path.join(imagenet_dir, src),
    #                 os.path.join(test_save_dir, dst))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagenet-dir", default='imagenet', type=str)
    parser.add_argument("--save-dir", default='imagenet50', type=str)
    parser.add_argument('--mode', type=str, default='50',
                        choices=['50', '300', '919', 'all'])
    args = parser.parse_args()

    if args.mode == 'all':
        make('50', args.imagenet_dir, args.save_dir)
        make('300', args.imagenet_dir, args.save_dir)
        make('919', args.imagenet_dir, args.save_dir)
    else:
        make(args.mode, args.imagenet_dir, args.save_dir)
