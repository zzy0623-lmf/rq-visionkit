# -*- coding: utf-8 -*-
"""M3 低代码部署控制台（T2.3）。

模块组成：
- pack.py   模型文件 + config.yaml → 部署包
- ssh.py    下发传输（本地 / OpenSSH subprocess）
- routes.py  /deploy 路由组（打包、下发、结果看板对接 runtime）
"""
