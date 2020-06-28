## 项目说明
通过Numpy,pandas来分析电影数据集，并用matplotlib对数据集可视化


	input2 = [label="input|size:|(299,299,3)"]
    a[label="ResNet50|{input:|output:}|{(224, 224, 3)|(7,7,2048)}"]
    b[label="VGG16|{input:|output:}|{(224, 224, 3)|(7,7,512)}"]
    c[label="Xception|{input:|output:}|{(299, 299, 3)|(10,10,2048)}"]
	GlobalAvgPooling=[label="GlobalAveragePooling"]
    Concatenate[label="Concatenate|output:|(None,4608)"]
    Dropout[label="Dropout|Rate:|0.5"]
    Dense[label="Dense|{input:|output:}|{(4608)|(1)}"]
    input1 -> a -> GlobalAvgPooling -> Concatenate
    input1 -> b -> GlobalAvgPooling -> Concatenate
    input2 -> c -> GlobalAvgPooling -> Concatenate
    Concatenate -> Dropout -> Output