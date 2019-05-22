## 项目介绍
 通过命令行执行python脚本，对花用pytorch框架预测花卉种类
 
### 使用
  第一个文件 train.py 将用数据集训练新的网络，并将模型保存为检查点。第二个文件 predict.py 将使用训练的网络预测输入图像的类别
  * 用 train.py 用数据集训练新的网络
    * 基本用途：python train.py data_directory
    * 在训练网络时，输出训练损失、验证损失和验证准确率
    * 选项：
      * 设置保存检查点的目录：python train.py data_dir --save_dir save_directory
      * 选择架构：python train.py data_dir --arch "vgg16"
      * 设置超参数：
        * python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 100
        * 使用 GPU 进行训练：python train.py data_dir --gpu
    
   * 使用 predict.py 预测图像的花卉名称以及该名称的概率
     * 基本用法：python predict.py input checkpoint
     * 选项:
       * 返回前K个类别：python predict.py input checkpoint --top_k 5
       * 使用类别到真实名称的映射: python predict.py input checkpoint --category_names cat_to_name.json
       * 使用 GPU 进行训练：python predict.py input checkpoint --gpu
