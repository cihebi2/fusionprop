# filepath: C:/Users/ciheb/Desktop/AA_solubility/web/protein_webapp/protein_webapp/celery.py
import os
import sys
from pathlib import Path

# This celery.py file is expected to be at AA/fusionprop/protein_webapp/celery.py
# We want to add AA/fusionprop to the sys.path
# project_root should be the path to 'AA/fusionprop'
project_root = Path(__file__).resolve().parent.parent

# Ensure the path is added only once and at the beginning
if str(project_root) not in sys.path or sys.path[0] != str(project_root):
    # Remove if it exists elsewhere, to ensure it's first
    if str(project_root) in sys.path:
        sys.path.remove(str(project_root))
    sys.path.insert(0, str(project_root))

# print(f"DEBUG SYS_PATH in AA/fusionprop/protein_webapp/celery.py: {{sys.path}}") # Uncomment for debugging

# 设置 Django 的 settings 模块
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'protein_webapp.settings')

from celery import Celery

app = Celery('protein_webapp')

# 使用 Django settings 配置 Celery
# namespace='CELERY' 表示所有 Celery 配置键都应以 `CELERY_` 开头
app.config_from_object('django.conf:settings', namespace='CELERY')

# 自动发现各个 app 下的 tasks.py 文件
app.autodiscover_tasks()

@app.task(bind=True)
def debug_task(self):
    print(f'Request: {self.request!r}')