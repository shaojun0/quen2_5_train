import time
import torch
import os
import shutil
import gc
import random
import numpy as np
import deepspeed
from deepspeed.accelerator import get_accelerator
from save_model_utils import get_model, validate_arguments, parse_arguments
from torch_save_utils import load_io_ops

def _get_ds_config(args, writer_type, use_gds):
    ds_config = {
        "train_micro_batch_size_per_gpu": 1,
        "zero_optimization": {
            "stage": args.zero_stage,
            "cpu_offload": args.cpu_offload
        },
        "fp16": {
            "enabled": args.half
        },
        "optimizer": {
            "type": "Adam",
            "params": {
                "torch_adam": not args.fused
            }
        },
        "checkpoint": {
            "checkpoint_serialization": args.zipfile 
        },
        "aio": {
            "block_size": 8 * (1024**2),
            "queue_depth": 8,
            "single_submit": False,
            "overlap_events": True,
            "intra_op_parallelism": 2,
            "use_gds": use_gds,
        }
    }

    if writer_type:
        ds_config["checkpoint"]["writer"] = {
            "type": writer_type,
            "io_buffer_size": args.io_buffer_mb * (1024**2),
            "io_buffer_double": not args.single_io_buffer,
            "show_statistics": not args.no_statistics,
            "data_parallel": "socket" #   None # not args.single_writer
        }

    return ds_config


def _get_ds_engine(model, ds_config):
    ds_engine, _, _, _ = deepspeed.initialize(
        model=model, model_parameters=model.parameters(), config=ds_config)

    return ds_engine


def _do_optimizer_step(ds_engine):
    for p in ds_engine.module.parameters():
        p.grad = torch.zeros_like(p)
    ds_engine.step()


def _free_ds_memory(ds_engine):
    ds_engine.destroy()
    del ds_engine
    ds_engine = None
    gc.collect()
    get_accelerator().empty_cache()


def test_save(tag, folder, model, args, writer_type):
    use_gds = writer_type == 'fast' and 'gds' in tag
    ds_config = _get_ds_config(args, writer_type, use_gds)
    ds_engine = _get_ds_engine(model, ds_config)
    if args.zero_stage == 0:
        _do_optimizer_step(ds_engine)

    import pdb; pdb.set_trace()
    st = time.time()
    ds_engine.save_checkpoint(save_dir=folder, tag=tag)
    write_sec = time.time() - st
    import pdb; pdb.set_trace()
    _free_ds_memory(ds_engine)
    return write_sec


def _get_folder_size(folder):
    size = 0
    for path, _, files in os.walk(folder):
        size += sum([os.path.getsize(os.path.join(path, f)) for f in files])
    return size


def run(model, model_name, ckpt_name, args):
    print(f'Model name = {model_name}')
    writer_dict = {
        'test_save': None,
        'test_ds_mock_save': 'mock',
        'test_ds_py_save': 'python',
        'test_ds_aio_fast_save': 'fast',
        'test_ds_gds_fast_save': 'fast',
    }
    for tag, writer_type in writer_dict.items():
        folder = os.path.join(args.folder, ckpt_name, tag)
        if os.path.exists(folder):
            shutil.rmtree(folder, ignore_errors=True)
        write_sec = test_save(tag, folder, model, args, writer_type)
        ckpt_size = _get_folder_size(folder)
        gb_size = ckpt_size / (1024**3)
        gb_per_sec = gb_size / write_sec
        print(
            f'{tag} -- {gb_size:5.2f} GB, {write_sec:5.2f} secs, {gb_per_sec:5.2f} GB/s'
        )
        print(f'*********************************************')

def init_torch_distributed():
    import torch.distributed as dist
    from deepspeed.constants import TORCH_DISTRIBUTED_DEFAULT_PORT, CROSS_RANK, CROSS_SIZE
    os.environ['MASTER_PORT'] = str(TORCH_DISTRIBUTED_DEFAULT_PORT)
    os.environ['MASTER_ADDR'] = "localhost"
    os.environ['LOCAL_RANK'] = str(0)
    os.environ['WORLD_SIZE'] = str(1)
    os.environ['CROSS_RANK'] = str(0)
    os.environ['CROSS_SIZE'] = str(1)
    dist.init_process_group(backend='nccl', rank=0, world_size=1)



def main():
    print(
        f'Performance test of deepspeed integration of fast model checkpointing.'
    )
    print(f'torch version = {torch.__version__}')
    init_torch_distributed()
    torch.manual_seed(42)
    np.random.seed(0)
    random.seed(0)
    args = parse_arguments()
    if not validate_arguments(args):
        quit()
    load_io_ops(args)
    model, model_name, ckpt_name = get_model(args.model)
    run(model, model_name, ckpt_name, args)


if __name__ == "__main__":
    main()
