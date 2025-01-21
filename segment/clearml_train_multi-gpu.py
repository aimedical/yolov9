import os
import sys
import argparse
from pathlib import Path
from torch.distributed.launcher.api import LaunchConfig, elastic_launch
from torch.distributed.elastic.multiprocessing.errors import record
from clearml import Dataset, Task
import runpy

# ここで ROOT を定義（例：スクリプトのディレクトリを基準にする）
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def training_func(args):
    # train_with_trainer.py を呼び出すスクリプトパス
    # 必要に応じて models/ 配下の構成に合わせて変更してください
    train_script = './clearml_train.py'

    # 以下のように、置き換えた引数を sys.argv に渡す
    # 不要なものはコメントアウトするなど調整してください
    sys.argv = [
        train_script,
        '--weights', str(args.weights),
        '--cfg', str(args.cfg),
        '--data', str(args.data),
        '--hyp', str(args.hyp),
        '--epochs', str(args.epochs),
        '--batch-size', str(args.batch_size),
        '--imgsz', str(args.imgsz),
        '--optimizer', str(args.optimizer),
        '--workers', str(args.workers),
        '--project', str(args.project),
        '--name', str(args.name),
        f'--local_rank={args.local_rank}',  # DDP用ローカルランク

        # boolフラグ系やオプション引数
        *(('--rect',) if args.rect else ()),
        *(('--resume',) if args.resume else ()),
        *(('--nosave',) if args.nosave else ()),
        *(('--noval',) if args.noval else ()),
        *(('--noautoanchor',) if args.noautoanchor else ()),
        *(('--noplots',) if args.noplots else ()),
        *(('--evolve', str(args.evolve)) if args.evolve else ()),
        *(('--cache', args.cache) if args.cache else ()),
        *(('--image-weights',) if args.image_weights else ()),
        *(('--multi-scale',) if args.multi_scale else ()),
        *(('--single-cls',) if args.single_cls else ()),
        *(('--sync-bn',) if args.sync_bn else ()),
        *(('--exist-ok',) if args.exist_ok else ()),
        *(('--quad',) if args.quad else ()),
        *(('--cos-lr',) if args.cos_lr else ()),
        *(('--label-smoothing', str(args.label_smoothing)) if args.label_smoothing else ()),
        *(('--patience', str(args.patience)) if args.patience else ()),
        *(('--freeze', *map(str, args.freeze)) if args.freeze else ()),
        *(('--save-period', str(args.save_period)) if args.save_period > 0 else ()),
        *(('--seed', str(args.seed)) if args.seed else ()),
        *(('--close-mosaic', str(args.close_mosaic)) if args.close_mosaic else ()),
        *(('--mask-ratio', str(args.mask_ratio)) if args.mask_ratio else ()),
        *(('--no-overlap',) if args.no_overlap else ()),
        # ClearML 用（train_with_trainer.py が参照するなら渡す）
        *(('--dataset_id', args.dataset_id) if args.dataset_id else ()),
        *(('--dataset_path', args.dataset_path) if args.dataset_path else ()),
        *(('--model_id', args.model_id) if args.model_id else ()),
    ]

    # training_func のあるディレクトリを PYTHONPATH に追加
    train_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, train_dir)

    # train.py を実行（ファイル内の __main__ として実行する）
    runpy.run_path(train_script, run_name='__main__')


@record
def main():
    # ClearML タスクの初期化
    task = Task.init(project_name='ssl_exp', task_name='rtdetrv2_multigpu_test')

    parser = argparse.ArgumentParser(description='Distributed Training Script for ClearML')

    # -------------------------------------------------------------
    # 置き換えた引数群 (YOLO などの学習スクリプトで使う想定)
    # -------------------------------------------------------------
    parser.add_argument('--weights', type=str, default=ROOT / 'yolo-seg.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128-seg.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch-low.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=100, help='total training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable AutoAnchor')
    parser.add_argument('--noplots', action='store_true', help='save no plot files')
    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='image --cache ram/disk')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW', 'LION'], default='SGD', help='optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', default=ROOT / 'runs/train-seg', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--cos-lr', action='store_true', help='cosine LR scheduler')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone=10, first3=0 1 2')
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--seed', type=int, default=0, help='Global training seed')
    parser.add_argument('--close-mosaic', type=int, default=0, help='Experimental')
    # Instance Segmentation
    parser.add_argument('--mask-ratio', type=int, default=4, help='Downsample the truth masks to saving memory')
    parser.add_argument('--no-overlap', action='store_true', help='Overlap masks train faster at slightly less mAP')

    # ClearML で使用するための追加引数例（必要に応じて変更）
    parser.add_argument("--dataset_id", type=str, default="e003ffafa61547d3bf517df1e72712af")
    parser.add_argument("--dataset_path", type=str, default="/coco")
    parser.add_argument("--model_id", type=str)

    # -------------------------------------------------------------
    # 以下は環境設定・分散学習向け引数など
    # -------------------------------------------------------------
    parser.add_argument('--print-method', type=str, default='builtin', help='Print method')
    parser.add_argument('--print-rank', type=int, default=0, help='Rank to print from')
    # local rank (ClearML / torchrun が自動で渡すことを想定)
    parser.add_argument('--local-rank', type=int, default=int(os.environ.get('LOCAL_RANK', 0)), help='Local rank ID')
    # 分散学習用
    parser.add_argument('--nproc_per_node', type=int, default=2, help='Number of processes per node')
    parser.add_argument('--nnodes', type=int, default=1, help='Number of nodes')
    parser.add_argument('--node_rank', type=int, default=0, help='Node rank')
    parser.add_argument('--master_addr', type=str, default='127.0.0.1', help='Master address')
    parser.add_argument('--master_port', type=str, default='29500', help='Master port')

    args = parser.parse_args()

    # CUDA_VISIBLE_DEVICES の設定（必要に応じて変更）
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, range(args.nproc_per_node)))

    # LaunchConfig 作成
    config = LaunchConfig(
        min_nodes=args.nnodes,
        max_nodes=args.nnodes,
        nproc_per_node=args.nproc_per_node,
        run_id='training_run',
        rdzv_backend='static',
        rdzv_endpoint=f"{args.master_addr}:{args.master_port}",
        rdzv_configs={'rank': args.node_rank}
    )

    # 分散トレーニング開始
    elastic_launch(config, training_func)(args)


if __name__ == '__main__':
    main()