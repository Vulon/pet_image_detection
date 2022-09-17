import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from dataset import SegmentationDataset
from torch.utils.tensorboard import SummaryWriter
from transformers.integrations import TensorBoardCallback
from transformers.trainer_callback import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)


class TensorboardImageLogger(TensorBoardCallback):
    def __init__(
        self,
        model: torch.nn.Module,
        dataset: SegmentationDataset,
        indices_to_print: list[int],
    ) -> None:
        super().__init__()
        self.model = model
        self.dataset = dataset
        self.indices_to_print = indices_to_print

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        super().on_evaluate(args, state, control, **kwargs)

        # def on_epoch_end(
        #     self,
        #     args: TrainingArguments,
        #     state: TrainerState,
        #     control: TrainerControl,
        #     **kwargs
        # ):
        #     """
        #     Event called at the end of an epoch.
        #     """
        if self.tb_writer is None:
            self._init_summary_writer(args)
        print()
        print("Callback. epoch", state.epoch, "step", state.global_step)
        self.model.eval()
        fig, axes = plt.subplots(
            len(self.indices_to_print), 3, figsize=(6, 2 * len(self.indices_to_print))
        )
        fig.tight_layout(pad=0)
        for index, i in enumerate(self.indices_to_print):
            transformed = self.dataset[i]
            raw_image = self.dataset.get_raw_image(i)
            raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
            with torch.no_grad():
                image_data = (
                    transformed["image"].unsqueeze(0).to(self.model.model.device)
                )
                output = self.model(image=image_data)[0]
                output = output.cpu().detach().numpy()
                del image_data

            dogs_mask = output[0, :, :, 0]
            cats_mask = output[0, :, :, 1]

            axes[index][0].imshow(raw_image)
            axes[index][0].axis("off")
            axes[index][0].margins(0)
            axes[index][1].imshow(cats_mask)
            axes[index][1].axis("off")
            axes[index][1].margins(0)
            axes[index][2].imshow(dogs_mask)
            axes[index][2].axis("off")
            axes[index][2].margins(0)

        fig.canvas.draw()
        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(
            fig.canvas.get_width_height()[::-1] + (3,)
        )

        self.tb_writer.add_image(
            "plot",
            img_tensor=image_from_plot,
            global_step=state.global_step,
            dataformats="HWC",
        )

        self.model.train()
