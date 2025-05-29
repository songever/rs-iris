use serde::Deserialize;
use std::error::Error;
use std::fs::File;
use std::path::Path;


#[derive(Debug, Deserialize)]
pub struct IrisRecord {
    pub sepal_length: f64,
    pub sepal_width: f64,
    pub petal_length: f64,
    pub petal_width: f64,
    pub species: String,
}

impl IrisRecord {
    pub fn into_feature_vector(&self) -> Vec<f64> {
        vec![
            self.sepal_length,
            self.sepal_width,
            self.petal_length,
            self.petal_width,
        ]
    }
    pub fn into_label(&self) -> String {
        self.species.clone()
    }
}


pub fn load_iris_data<P: AsRef<Path>>(path: P) -> Result<Vec<IrisRecord>, Box<dyn Error>> {
    let file = File::open(path)?;
    let mut rdr = csv::Reader::from_reader(file);
    let mut records = Vec::new();

    for result in rdr.deserialize() {
        let record: IrisRecord = result?;
        records.push(record);
    }

    Ok(records)
}