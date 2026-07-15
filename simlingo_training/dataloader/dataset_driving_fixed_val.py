"""Deterministic validation wrapper for SimLingo's ordinary driving dataset.

Training behavior is inherited unchanged from ``Data_Driving``. During
validation, only the language task prompt is replaced with one fixed waypoint
prediction prompt. Images, waypoints, routes, target points, and all other
fields still come from the original dataset implementation.
"""

from simlingo_training.dataloader.dataset_driving import Data_Driving


class Data_Driving_FixedVal(Data_Driving):
    """Keep training randomization while making validation prompts repeatable."""

    def __getitem__(self, index):
        sample = super().__getitem__(index)

        if self.split == "train":
            return sample

        # Use target-point navigation deterministically during validation.
        # The parent dataset has already populated the matching placeholder
        # values, so both <TARGET_POINT> tokens are replaced as usual.
        prompt = (
            f"Current speed: {round(float(sample.speed), 1)} m/s. "
            "Target waypoint: <TARGET_POINT><TARGET_POINT>. "
            "Predict the waypoints."
        )
        answer = "Waypoints:"

        conversation_answer = [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer}],
            }
        ]
        conversation_all = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image"},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer}],
            },
        ]

        return sample._replace(
            conversation=conversation_all,
            answer=conversation_answer,
        )
