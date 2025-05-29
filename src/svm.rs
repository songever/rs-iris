use linfa::composing::MultiClassModel;
use linfa::prelude::*;
use linfa_svm::Svm;
use iris_data::IrisRecord;
use ndarray::Array2;
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

    let (train, valid) = dataset.shuffle(&mut rand::thread_rng())
        .split_with_ratio(0.7);

    println!(
        "Fit SVM classifier with #{} training points",
        train.nsamples()
    );

    let params = Svm::<_, Pr>::params()
        //.pos_neg_weights(5000., 500.)
        .gaussian_kernel(1.0);

    let model = train
        .one_vs_all()?
        .into_iter()
        .map(|(l, x)| (l, params.fit(&x).unwrap()))
        .collect::<MultiClassModel<_, _>>();

    let pred = model.predict(&valid);

    // create a confusion matrix
    let cm = pred.confusion_matrix(&valid)?;

    // Print the confusion matrix
    println!("{:?}", cm);

    // Calculate the accuracy and Matthew Correlation Coefficient (cross-correlation between
    // predicted and targets)
    println!("accuracy {}, MCC {}, recall {}", cm.accuracy(), cm.mcc(), cm.recall());

    Ok(())
}