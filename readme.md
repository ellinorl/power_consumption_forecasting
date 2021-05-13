Copyright 2021 Ludvig Eriksson, Jakob Johansson, Richard Johnsson, Lasse Kötz, Johan Lamm, Ellinor Lundblad.

This project is built for our bachelors thesis project at Chalmers University of Technology with the goal of creating 
AI-tools for prediction of future consumption. The project consists of three models, ANN, LSTM and Linear Regression.
The models and a prediction tool are easily accessible in AI_tool.py.

In our project, we have done some data preparation and cleaning. This code can be found under Data/Test_data.py and in
some separate functions for the models in ANN/ANN_clean.py, RNN/RNN_clean.py as well as 
LinearRegression/linear_regression_clean.py. Please note that the data preparation and cleaning process is done on 
our particular data and will likely not work for other datasets unless it is formatted in the same manner. An example of
our data can be found in Data/example_randomly_generated_day.csv. The data processing code is kept in the repo in case needed
for inspiration or further use but is not intended to work on any data except the one we used. **Our particular data is 
not included in the public project for privacy reasons.** 

Note that the Results/ file and its content is also kept for inspiration and further use but cannot be run without
data, which cannot be included in this public project. Note also that the code in the class files 'ANNModel' and
'LSTM_clean' are quite different, and for better ease of use, should be restructured to use the same methods
with the same inputs and similar structure.

**Please take care when using or changing any code that is not included in the final tool, AI_tool.py.**

With that said, here are some guidelines for our AI_tool.py file.

###Using the AI_tool.py file
***The AI_tool.py file contains 5 main functions:***
* train_model()
* model_prediction_next_24h()
* save_model()
* load_model()
* load_scalers_RNN()

To predict the coming 24h on a 15-minute basis with any of our models, please follow these steps.
1. Ensure that you have 24 full hours of data on the format as given in Data/example_randomly_generated_day.csv.
2. Ensure that you have your preferred model saved. Our final models are saved as final_model_ANN, final_model_RNN, and
final_model_LR.joblib and can be used.
3. Use the ```load_model()``` function with the path to your model. Example: 
   ```my_model = load_model(model_path="final_model_LR.joblib", model_type="LR")```
4. Then use the model returned from the ```load_model()``` as an input to ```model_prediction_next_24h()```:
   ```model_prediction_next_24h(model_prediction_next_24h(model=my_model,to_predict_path="Data/example_randomly_generated_day.csv", model_type="LR"))```

Note that ```model_path, model_type, ``` and ```to_predict_path``` can be set to what you prefer. The 
```example_randomly_generated_day.csv``` is randomly generated and provided for explanatory purposes only.

The train_model() and save_model() functions can and should be used to train and save new models when introduced to new
datasets. In this case, make sure that the data preparation functions in Data/Test_data and in RNN/RNN_clean.py, 
ANN/ANN_clean.py as well as LinearRegression/linear_regression_clean.py are updated to work with your dataset.





