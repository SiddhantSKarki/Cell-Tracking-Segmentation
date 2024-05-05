
import sys
import argparse
from skimage.io import imread
import os
import numpy as np
from skimage.measure import label, regionprops
import fnmatch 
import re

class JaccardEvaluator:
    def __init__(self):
        self.jaccard_scores = []

    def add_image_pair(self, seg_image_path, gt_image_path):
        seg_image = imread(seg_image_path) > 0
        gt_image = imread(gt_image_path)
        jaccard_index, indices = self.compute_jaccard_index_for_matches(gt_image, seg_image)
        self.jaccard_scores += list(indices.values())
        return jaccard_index, indices

    def compute_jaccard_index_for_matches(self, ref_image, seg_mask):
        labeled_mask = label(seg_mask)
        mask_props = regionprops(labeled_mask)
        jaccard_indices = {}

        for ref_label in np.unique(ref_image):
            if ref_label == 0:
                continue
            ref_object = (ref_image == ref_label)
            overlaps = []

            for prop in mask_props:
                intersection = np.sum(ref_object & (labeled_mask == prop.label))
                union = np.sum(ref_object | (labeled_mask == prop.label))
                if intersection > 0.5 * np.sum(ref_object):
                    jaccard_index = intersection / union
                    overlaps.append(jaccard_index)
                else:
                    overlaps.append(0)

            if overlaps:
                best_match_jaccard = max(overlaps)
                jaccard_indices[ref_label]=best_match_jaccard

        mean_jaccard_index = np.mean(list(jaccard_indices.values())) if jaccard_indices else 0
        return mean_jaccard_index, jaccard_indices

    def mean_jaccard_index(self):
        return np.mean(self.jaccard_scores)

    def report(self):
        return self.jaccard_scores

    @staticmethod
    def evaluate_folder(gt_folder, output_folder, image_patterns=['*.tif', '*.jpg', '*.png'], verbose=False):
        evaluator = JaccardEvaluator()
        all_files = os.listdir(gt_folder)
        gt_files = []
        for pattern in image_patterns:
            gt_files.extend(fnmatch.filter(all_files, pattern))

        for gt_file in sorted(gt_files):
            if verbose:
              print('-'*10)
              print(gt_file)
            match = re.search(r'man_seg(\d+)', gt_file)
            if match:
                file_number = match.group(1)
                output_file = f'mask{file_number}.tif'
                gt_path = os.path.join(gt_folder, gt_file)
                output_path = os.path.join(output_folder, output_file)
                if os.path.exists(output_path):
                    jac, indices = evaluator.add_image_pair(output_path, gt_path)
                    if verbose: 
                      print('JAC:',jac)
                      for idx, ijac in indices.items():
                        print('   ', idx, ':', ijac)

        return evaluator.mean_jaccard_index()


def main():
    parser = argparse.ArgumentParser(description="Evaluate segmentation results using Jaccard Index")
    parser.add_argument('--verbose', '-v', action='store_true', help='Output a lot of info.')
    parser.add_argument('output_folder', type=str, help='Folder containing output segmentation images')
    parser.add_argument('gt_folder', type=str, help='Folder containing ground truth segmentation images')
    args = parser.parse_args()

    jaccard_mean = JaccardEvaluator.evaluate_folder(args.output_folder, args.gt_folder, verbose=args.verbose)
    print(f"Mean Jaccard Index: {jaccard_mean}")

if __name__ == '__main__':
    main()
