# 使用官方Python 3.10镜像作为基础镜像
FROM python:3.10-slim

# 设置容器内的工作目录
WORKDIR /app

# 将当前目录下的所有文件复制到容器的工作目录
COPY . /app

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 设置容器启动时执行的命令
CMD ["python3", "app.py"]
