import os
import json

import torch
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from .datamodule import FederatedCIFAR100DataModule
from .model import DinoClassifier
from .federated import FedAvg


@hydra.main(version_base=None, config_path="../configs", config_name="fedavg")
def train(cfg: DictConfig) -> None:
    print("config:")
    print(OmegaConf.to_yaml(cfg))
    
    # hydra changes root dir
    original_cwd = get_original_cwd()
    data_dir = os.path.join(original_cwd, cfg.data.data_dir)
    checkpoint_dir = os.path.join(original_cwd, cfg.logging.checkpoint_dir)
    output_dir = os.path.join(original_cwd, cfg.logging.output_dir)
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(cfg.seed)
    
    datamodule = FederatedCIFAR100DataModule(
        data_dir=data_dir,
        num_clients=cfg.federated.num_clients,
        sharding=cfg.federated.sharding,
        num_classes_per_client=cfg.federated.num_classes_per_client,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        val_split=cfg.data.val_split,
        seed=cfg.seed,
        image_size=cfg.model.image_size,
    )
    datamodule.prepare_data()
    datamodule.setup()
    
    model = DinoClassifier(
        num_classes=cfg.model.num_classes,
        freeze_backbone=cfg.model.freeze_backbone,
    )

    if cfg.federated.use_sparse:
        print('init ridge')
        model.initialize_head_with_ridge(datamodule.train_dataloader(), device)

    print("training")
    fedavg = FedAvg(
        model=model,
        datamodule=datamodule,
        num_rounds=cfg.federated.num_rounds,
        participation_rate=cfg.federated.participation_rate,
        local_steps=cfg.federated.local_steps,
        lr=cfg.optimizer.lr,
        momentum=cfg.optimizer.momentum,
        weight_decay=cfg.optimizer.weight_decay,
        use_sparse=cfg.federated.use_sparse,
        sparsity_level=cfg.federated.sparsity_level,
        num_calibration_rounds=cfg.federated.num_calibration_rounds,
        sparse_strategy=cfg.federated.sparse_strategy,
        alpha=cfg.federated.alpha

    )
    
    # check for resume
    resume_from = cfg.get('resume_from', None)
    if resume_from:
        if not os.path.isabs(resume_from):
            resume_from = os.path.join(original_cwd, resume_from)
        start_round = fedavg.load_checkpoint(resume_from)
        print(f"resumed from round {start_round}")
    
    print("starting training...")
    checkpoint_path = os.path.join(checkpoint_dir, cfg.experiment_name)
    history = fedavg.fit(
        eval_every=cfg.logging.eval_every,
        checkpoint_path=checkpoint_path,
    )

    print("Evaluating on test set...")
    test_loss, test_acc = fedavg.evaluate_test()
    
    # save results
    results = {
        'config': OmegaConf.to_container(cfg),
        'history': history,
        'final_val_acc': history['val_acc'][-1],
        'final_test_loss': test_loss,
        'final_test_acc': test_acc,
    }
    
    results_path = os.path.join(output_dir, f"{cfg.experiment_name}.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # save final model
    model_path = os.path.join(output_dir, f"{cfg.experiment_name}_model.pt")
    torch.save(fedavg.global_model.state_dict(), model_path)
    
    print("training complete")
    print(f"final val_acc: {history['val_acc'][-1]}")
    print(f"final test_acc: {test_acc}")
    print(f"results saved to {results_path}")


if __name__ == "__main__":
    train()