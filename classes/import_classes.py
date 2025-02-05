import os
import pandas as pd
from typing import List, Dict
from typing import List, Optional, Tuple
import numpy as np
from dataclasses import dataclass, field, replace
from collections import defaultdict
import random
from itertools import cycle
from .media_object import MediaObject,MultimediaObject

# Perturbation-relevant metrics, see computed_on below
PERTURBATION_ORIGINAL: str = "original"
PERTURBATION_PERTURBED: str = "perturbed"
PERTURBATION_WORST: str = "worst"
CORRECT_TAG: str = "correct"
    
@dataclass(frozen=True)
class PerturbationDescription:
    """DataClass used to describe a Perturbation"""

    name: str
    """Name of the Perturbation"""

    robustness: bool = False
    """Whether a perturbation is relevant to robustness. Will be used to aggregate perturbations metrics"""

    fairness: bool = False
    """Whether a perturbation is relevant to fairness. Will be used to aggregate perturbations metrics"""

    computed_on: str = PERTURBATION_PERTURBED
    """Which types of Instances we are evaluating, to be populated during metric evaluation. PERTURBATION_PERTURBED
    (default) means we are evaluating on perturbed instances, PERTURBATION_ORIGINAL means we are evaluating the
    unperturbed version of instances where this perturbation applies, and, PERTURBATION_WORST means the the minimum
    metric between the two."""

    seed: Optional[int] = None
    """Seed added to instance_id when generating perturbation"""
    
@dataclass(frozen=True)
class Input:
    """
    The input of an `Instance`.
    """

    text: str = ""
    """The text of the input (e.g, passage to summarize, text for sentiment analysis, etc.)"""

    multimedia_content: Optional[MultimediaObject] = None
    """A single input can consists of multimodal content interleaved (e.g., text, image, text, ...)."""
    
@dataclass(frozen=True)
class Output:
    """
    The output of a `Reference`.
    """

    text: str = ""
    """The text of the output."""

    multimedia_content: Optional[MultimediaObject] = None
    """The output can be multimodal content interleaved (e.g., text, image, text, ...)."""
  
@dataclass(frozen=True)
class Reference:
    """
    A `Reference` specifies a possible output and how good/bad it is.  This
    could be used to represent multiple reference outputs which are all
    acceptable (e.g., in machine translation) or alternatives (e.g., in a
    multiple-choice exam).
    """

    output: Output
    """The output"""

    tags: List[str]
    """Extra metadata (e.g., whether it's correct/factual/toxic)"""

    @property
    def is_correct(self) -> bool:
        return CORRECT_TAG in self.tags

    def render_lines(self) -> List[str]:
        return [f"reference {format_tags(self.tags)}: {format_text(self.output.text)}"]

@dataclass(frozen=True, eq=False)
class Instance:
    """
    An `Instance` represents one data point that we're evaluating on (e.g., one
    question in a QA task).
    Note: `eq=False` means that we hash by the identity.
    """

    input: Input
    """The input"""

    references: List[Reference]
    """References that helps us evaluate"""

    split: Optional[str] = None
    """Split (e.g., train, valid, test)"""

    sub_split: Optional[str] = None
    """Sub split (e.g. toxic, non-toxic)"""

    id: Optional[str] = None
    """Used to group Instances that were created from a particular Instance through data augmentation"""

    perturbation: Optional[PerturbationDescription] = None
    """Description of the Perturbation that was applied when creating this Instance"""

    contrast_inputs: Optional[List[Input]] = None
    """Perturbed input as defined by contrast sets (if available)"""

    contrast_references: Optional[List[List[Reference]]] = None
    """References for the perturbed input above (if available)"""

    @property
    def first_correct_reference(self) -> Optional[Reference]:
        """Return the first correct reference."""
        for reference in self.references:
            if reference.is_correct:
                return reference
        return None

    @property
    def all_correct_references(self) -> List[Reference]:
        """Return all correct references."""
        return [reference for reference in self.references if reference.is_correct]

    def render_lines(self) -> List[str]:
        info = [f"input: {format_text(self.input.text)}"]
        if self.sub_split:
            info.append(f"sub_split: {format_text(self.sub_split)}")
        if self.id:
            info.append(f"id: {format_text(self.id)}")
        if self.perturbation:
            info.append(f"perturbation: {self.perturbation}")

        for reference in self.references:
            info.extend(reference.render_lines())

        return info
    
@dataclass(frozen=True)
class PassageQuestionInput(Input):
    """
    Passage-question pair used for question answering Scenarios.
    """

    def __init__(
        self,
        passage: str,
        question: str,
        passage_prefix: str = "",
        question_prefix: str = "Question: ",
        separator: str = "\n",
    ):
        super().__init__(f"{passage_prefix}{passage}{separator}{question_prefix}{question}")