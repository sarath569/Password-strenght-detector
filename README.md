# Password-strenght-detector

A python tool used to classify the given password as weak, medium and strong with the help of hugging face transform pre trained model

The model is trained with 3 classes strong,medium and weak each contains 3000 words at epoch of 3 and batch size of 32.

The model is trained in such a way that the weightage is in this order weak > medium > strong . This is achieved using custom loss function. 
So the model will be 3x sensitive to weak passsowrd, 2x sensitive to medium password, x sensitive to string password

Train the model with the following command:

python3.11 train_password_strenght.py classes/weak_passwords.txt classes/medium_passwords.txt classes/strong_passwords.txt

The weights will be stored in results directory after training.

To test the model, run:

python3.11 predict_password_strenght.py path_where_weight_stored "password" 
