use linfa::metrics::ToConfusionMatrix;
use linfa::traits::{Fit, Predict};
use linfa_bayes::GaussianNb;
use iris_data::IrisRecord;
use ndarray::Array2;
use linfa::dataset::Dataset;


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

    let (train, test) = dataset.shuffle(&mut rand::thread_rng())
        .split_with_ratio(0.7);
    // Train the model
    let model: GaussianNb<_, _> = GaussianNb::params().fit(&train)?;
    // Predict the validation dataset
    let pred = model.predict(&test);

    // Construct confusion matrix
    let cm = pred.confusion_matrix(&test)?;

    println!("{:?}", cm);
    println!("accuracy {}, MCC {}, recall{}", cm.accuracy(), cm.mcc(), cm.recall());

    Ok(())
}
