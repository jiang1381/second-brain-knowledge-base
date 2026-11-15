# Python 学习笔记

## 可变与不可变
list 可变，tuple 不可变；dict 的 key 必须是不可变类型。

## 生成器
用 yield 惰性产出数据，不一次性载入内存，适合处理大文件和数据流。

## 常用标准库
- pathlib：路径操作
- json：序列化
- argparse：命令行参数
- http.server：快速起一个 Web 服务
- urllib.request：发 HTTP 请求，调用大模型 API 就是它

## 从 Java 转 Python 的体会
Python 更简洁，但要小心缩进和动态类型带来的运行时错误。
