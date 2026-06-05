#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Batch runner for HoloClean baseline."""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--base_out', required=True)
    p.add_argument('--python', default=sys.executable)
    p.add_argument('--holoclean_home', default=None)
    p.add_argument('--auto_mine_fds', action='store_true')
    p.add_argument('--fd_confidence', type=float, default=0.98)
    p.add_argument('--fd_min_support', type=int, default=5)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--db_name', default='holo')
    p.add_argument('--db_user', default='holocleanuser')
    p.add_argument('--db_pwd', default='abcd1234')
    p.add_argument('--db_host', default='localhost')
    args = p.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding='utf-8'))
    base_out = Path(args.base_out); base_out.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve().parent / 'run_holoclean_baseline.py'
    for item in cfg:
        name = item['name']
        out_dir = base_out / f'{name}_holoclean'
        cmd = [args.python, str(script), '--name', name, '--dirty', item['dirty'], '--clean', item['clean'], '--out_dir', str(out_dir), '--epochs', str(args.epochs), '--threads', str(args.threads), '--fd_confidence', str(args.fd_confidence), '--fd_min_support', str(args.fd_min_support), '--db_name', args.db_name, '--db_user', args.db_user, '--db_pwd', args.db_pwd, '--db_host', args.db_host]
        if args.holoclean_home: cmd += ['--holoclean_home', args.holoclean_home]
        if args.auto_mine_fds: cmd += ['--auto_mine_fds']
        if item.get('fd_path'): cmd += ['--fd_path', item['fd_path']]
        if item.get('dc_path'): cmd += ['--dc_path', item['dc_path']]
        print('\n[RUN]', ' '.join(cmd))
        subprocess.run(cmd, check=True)
if __name__ == '__main__':
    main()
