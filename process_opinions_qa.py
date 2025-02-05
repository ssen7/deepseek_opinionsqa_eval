import os
import pandas as pd
from typing import List, Dict
from typing import List, Optional, Tuple
import numpy as np
from dataclasses import dataclass, field, replace
from collections import defaultdict
import random
from itertools import cycle

from classes.import_classes import *
from classes.adapter_spec import AdapterSpec
""" Reference tags """
CORRECT_TAG: str = "correct"
    
""" Data splits """
TRAIN_SPLIT: str = "train"
VALID_SPLIT: str = "valid"
TEST_SPLIT: str = "test"
EVAL_SPLITS: List[str] = [VALID_SPLIT, TEST_SPLIT]
ALL_SPLITS: List[str] = [TRAIN_SPLIT] + EVAL_SPLITS
ADAPTER_SPEC=AdapterSpec()

def read_survey_questions(csv_path):
    df = pd.read_csv(csv_path, sep="\t")
    df["options"] = df.apply(lambda x: eval(x["options"]), axis=1)
    return df


def return_prompt_instances(config):
    output_path=config['prompt']['output_path']
    context=config['prompt']['context']
    survey_type=config['prompt']['survey_type']
    
    # Read all the instances
    instances = []
    splits: Dict[str, str] = {
        "dev": TRAIN_SPLIT,
        "test": TEST_SPLIT,
    }

    all_splits = ["dev", "test"] if context == "steer-qa" else ["test"]
    csv_dict = {
        "dev": os.path.join(output_path, f"./data/model_input/{context}.csv"),
        "test": os.path.join(output_path, f"./data/model_input/{survey_type}.csv"),
    }
    
    bios_df = None
    if context in ["steer-bio", "steer-portray"]:
        bios_path = os.path.join(output_path, f"./data/model_input/{context}.csv")
        bios_df = pd.read_csv(bios_path, sep="\t")
        
    for split in all_splits:
        csv_path: str = csv_dict[split]
        assert os.path.exists(csv_path)

        question_df = read_survey_questions(csv_path)

        for qidx, (question, answers) in enumerate(zip(question_df["question"], question_df["options"])):
            # Opinions QA test questions have no correct answer and thus we set it to be None by default
            # for all test instances.
            # In the case where context = steer-qa, we add demographic information in the form of a
            # in-context question answer pair as shown in the example above.

            correct_answer = None if split == "test" else question_df["correct"][qidx]

            def answer_to_reference(answer: str):
                return Reference(
                    Output(text=answer),
                    tags=[CORRECT_TAG] if (answer == correct_answer and split != "test") else [],
                )

            if bios_df is None:
                # context = "default" or "steer-qa"
                instance = Instance(
                    Input(text=question),
                    references=list(map(answer_to_reference, answers)),
                    split=splits[split],
                )
                instances.append(instance)
            else:
                # context = "steer-bio"or "steer-portray"
                for bio in bios_df["question"].values:
                    context = PassageQuestionInput(passage=bio, question=question + "\n")
                    instance = Instance(
                        context,
                        references=list(map(answer_to_reference, answers)),
                        split=splits[split],
                    )
                    instances.append(instance)
                    
    return instances
    
    
def sample_examples(
        all_train_instances: List[Instance], seed: int, sample_train: bool = True
    ) -> List[Instance]:
        """
        Sample a random set of train instances to use as examples by following the steps below:
        1. Sort the class labels (i.e., correct References) by the number of Instances that belong to the
           class so more common labels are included in the in-context examples. Break ties by shuffling.
        2. Keep sampling one train Instance from each class in the order established in step 1, until
           there are k examples.
        3. If we run out of examples to sample, sample the rest from the Instances that do not have
           class labels.

        Example:

            If we had to sample 2 instances from these train instances:
                Instance("say no", references=[Reference("no", tags=[CORRECT_TAG])]),
                Instance("say yes", references=[Reference("yes", tags=[CORRECT_TAG])]),
                Instance("say yes", references=[Reference("yes", tags=[CORRECT_TAG])]),

            The following instances would be selected:

                Instance("say yes", references=[Reference("yes", tags=[CORRECT_TAG])])
                Instance("say no", references=[Reference("no", tags=[CORRECT_TAG])])

        Returns a new list of randomly sampled train instances.
        """
        # Fix the random seed for reproducibility
        random.seed(seed)
        num_instances_to_sample: int = min(len(all_train_instances), ADAPTER_SPEC.max_train_instances)

        examples: List[Instance] = []
        if not sample_train:
            # Select sequentially from the train set
            examples = all_train_instances[num_instances_to_sample * seed : num_instances_to_sample * (seed + 1)]
            return examples

        unlabeled_instances: List[Instance] = []
        label_to_instances: Dict[str, List[Instance]] = defaultdict(list)
        for instance in all_train_instances:
            if instance.first_correct_reference:
                label_to_instances[instance.first_correct_reference.output.text].append(instance)
            else:
                unlabeled_instances.append(instance)

        # Build Instance counts to labels
        instances: List[Instance]
        counts_to_labels: Dict[int, List[str]] = defaultdict(list)
        for label, instances in sorted(label_to_instances.items()):
            counts_to_labels[len(instances)].append(label)

        sorted_labels: List[str] = []
        # Sort the labels by the number of Instances that belong to them
        for count in sorted(counts_to_labels, reverse=True):
            labels: List[str] = counts_to_labels[count]
            # Break ties by randomly shuffling labels that have the same number of Instances
            random.shuffle(labels)
            sorted_labels.extend(labels)

        labels_iterable = cycle(sorted_labels)
        while num_instances_to_sample > 0:
            next_label: Optional[str] = next(labels_iterable, None)
            if not next_label:
                break

            instances = label_to_instances[next_label]
            # If there are no Instances to sample for this particular label, skip it.
            if len(instances) == 0:
                continue

            # Randomly sample without replacement
            examples.append(instances.pop(random.randrange(len(instances))))
            num_instances_to_sample -= 1

        # If we ran out of Instances with correct References, sample the rest from
        # the pool of Instances without any References
        examples += random.sample(unlabeled_instances, num_instances_to_sample)
        return examples

def get_prefix_char(prefix: str) -> str:
    return [char for char in prefix if char.isalnum()][0]


def get_reference_prefix(prefix: str, i: int) -> str:
    """
    Example: prefix = "\nA. ", i = 2, return "\nC. "
    """
    prefix_char = get_prefix_char(prefix)
    return prefix.replace(prefix_char, chr(ord(prefix_char) + i))


def construct_example_prompt(instance: Instance, include_output: bool, reference_index: Optional[int]) -> str:
    """Return a list of lines corresponding to this example (part of the prompt)."""
    # Input
    result: str = ADAPTER_SPEC.input_prefix + instance.input.text + ADAPTER_SPEC.input_suffix

    # Include the references
    delimiter = ", "
    no_correct_references = "n/a"
    prefix_char = get_prefix_char(ADAPTER_SPEC.reference_prefix)
    output = no_correct_references
    for reference_index, reference in enumerate(instance.references):
        prefix = get_reference_prefix(ADAPTER_SPEC.reference_prefix, reference_index)
        result += prefix + reference.output.text + ADAPTER_SPEC.reference_suffix
        if reference.is_correct:
            if output == no_correct_references:
                output = get_reference_prefix(prefix_char, reference_index)
            elif ADAPTER_SPEC.multi_label:
                output += delimiter
                output += get_reference_prefix(prefix_char, reference_index)

    if include_output:
        result += ADAPTER_SPEC.output_prefix + output + ADAPTER_SPEC.output_suffix
    else:
        result += ADAPTER_SPEC.output_prefix.rstrip()

    return result


def generate_prompt(config):
    # Instruction text
    instructions_block: str = ADAPTER_SPEC.instructions
    include_output=config['prompt']['include_output']
    reference_index=config['prompt']['reference_index']
    
    instances=return_prompt_instances(config)
    all_train_instances: List[Instance] = [instance for instance in instances if instance.split == TRAIN_SPLIT]

    # Pick out evaluation instances. This includes both valid and test splits.
    eval_instances: List[Instance] = [instance for instance in instances if instance.split in EVAL_SPLITS]

    # print(
    #     f"{len(instances)} instances, "
    #     f"choosing {len(all_train_instances)}/{len(all_train_instances)} train instances, "
    #     f"{len(eval_instances)} eval instances"
    # )
    # Text for in-context training instances
    train_instance_blocks: List[str] = [
        construct_example_prompt(inst, include_output=True, reference_index=reference_index) for inst in all_train_instances
    ]

    # Example text
    eval_instance_block: str = [
        construct_example_prompt(inst, include_output=include_output, reference_index=reference_index) for inst in eval_instances
    ]
    return train_instance_blocks, eval_instance_block

if __name__=='__main__':
    print(generate_prompt())