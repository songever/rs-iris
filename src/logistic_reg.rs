use iris_data::IrisRecord;
use ndarray::Array2;
use linfa::dataset::Dataset;
use linfa::prelude::*;
use linfa_logistic::{MultiLogisticRegression, MultiFittedLogisticRegression};

pub fn run() -> Result<(), Box<dyn std::error::Error>> {
    let data: Vec<IrisRecord> = iris_data::load_iris_data("./data/iris.csv")?;

    let n_samples = data.len();
    let n_features = 4;

    let features: Array2<f64> = Array2::from_shape_vec(
        (n_samples, n_features),
        data.iter()
            .flat_map(|r| r.into_feature_vector())  // 返回 Vec<f64>
            .collect::<Vec<f64>>(),
    )?;

    let labels: Vec<String> = data.iter()
        .map(|r| r.into_label()) // 返回 String 或 &str
        .collect();

    //创建 linfa 的 Dataset
    let dataset = Dataset::new(features, labels.into())
        .map_targets(|species| match species.as_str() {
            "Iris-setosa" => 1usize,
            "Iris-versicolor" => 2usize,
            "Iris-virginica" => 0usize,
            _ => panic!("Unknown species"),
    });
    //也可以使用 linfa_datasets 提供的 iris 数据集
    // use linfa_datasets::iris;
    // let dataset = iris(); 


    // 随机打乱并划分数据集
    let mut rng = rand::thread_rng();
    let (train, test) = dataset.shuffle(&mut rng).split_with_ratio(0.7);
    println!("Number of train set: {}", train.nsamples());
    println!("Number of test set: {}", test.nsamples());
    // 数据集的标签为 3 是多分类问题
    let n_classes = train.targets().to_owned().into_iter().collect::<std::collections::HashSet<_>>().len();
    println!("Number of classes: {}", n_classes);  // 应该是 3

    // 训练多分类逻辑回归模型
    let model: MultiFittedLogisticRegression<_, _> = MultiLogisticRegression::default()
        .max_iterations(50)
        .fit(&train)
        .unwrap();

    // 打印模型参数
    println!("Model parameters: {:?}", model.params());

    // 使用模型进行预测
    let pred = model.predict(&test);
    println!("Predictions: {:?}", test.targets());
    println!("Predictions: {:?}", pred);
    // 计算混淆矩阵
    let cm = pred.confusion_matrix(&test).unwrap();

    // Print the confusion matrix, this will print a table with four entries. On the diagonal are
    // the number of true-positive and true-negative predictions, off the diagonal are
    // false-positive and false-negative
    println!("{:?}", cm);
    // Calculate the accuracy and Matthew Correlation Coefficient (cross-correlation between
    // predicted and targets)
    println!("accuracy {}, MCC {}, recall {}", cm.accuracy(), cm.mcc(), cm.recall());
    Ok(())
}
