一个全局替换脚本；一个补丁（1. paddle不存在的算子  2.paddle存在功能不一致，用全局替换脚本替换）。

全局替换脚本：

 1. 全局替换固定内容：   

    ```
    torch
    paddle
    torch.utils.data.Dataset  -> io.Dataset
    ```

    

    还要记录替换行数

    