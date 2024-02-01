from collections import Counter
import random

def random_shuffle_modalities_with_fixed_shifts(n_blocks_per_modality, n_shifts_per_run):
    def solution_candidate(n_blocks, n_shifts):
        n_repeats = n_blocks - 1 - n_shifts
        templates = ['repeat'] * n_repeats + ['shift'] * n_shifts
        random.shuffle(templates)
        templates.insert(0, '_')

        candidate = []
        for i, template in enumerate(templates):
            if i == 0:
                candidate.append(random.choice(['image', 'text']))
                continue

            if template == 'repeat':
                candidate.append(candidate[-1])
            elif template == 'shift':
                prev = candidate[-1]
                if prev == 'image':
                    candidate.append('text')
                elif prev == 'text':
                    candidate.append('image')
        return candidate
    
    while True:
        candidate = solution_candidate(2 * n_blocks_per_modality, n_shifts_per_run)
        candidate_counter = Counter(candidate)

        if candidate_counter['image'] == 5 and candidate_counter['text'] == 5:
            return candidate


def random_shuffle_modalities(n_blocks_per_modality):
    def solution_candidate():
        template = ['image'] * n_blocks_per_modality + ['sound'] * n_blocks_per_modality + ['text'] * n_blocks_per_modality + ['none'] * n_blocks_per_modality
        random.shuffle(template)
        return template
    
    while True:
        candidate = solution_candidate()

        no_duplicates = True
        for i in range(len(candidate) - 1):
            if candidate[i] == candidate[i + 1]:
                no_duplicates = False
                break

        if no_duplicates:
            return candidate

def random_sample_coco_id(coco_id_list, n_total_trials):
    sampled_coco_id_list = [] # list of [image, sound, text]
    index = 0
    while index < n_total_trials:
        while True:
            sample = random.sample(coco_id_list, 3)

            images = [pair[0] for pair in sampled_coco_id_list]
            sounds = [pair[1] for pair in sampled_coco_id_list]
            texts = [pair[2] for pair in sampled_coco_id_list]

            if (sample[0] not in images) and (sample[1] not in sounds) and (sample[2] not in texts):
                sampled_coco_id_list.append(sample)
                index += 1
                break
    
    return sampled_coco_id_list