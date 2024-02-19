import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import models,layers
#Récuperer la base de données(bas d'entrainement+test)
def read_efi_binary_time_series_vx(myfiles):
    dt = np.dtype([('frequences', 'f4'), ('ux', 'f4'), ('uy', 'f4'), ('uz', 'f4'), ('vx', 'f4'), ('vy', 'f4'), ('vz', 'f4'), ('ax', 'f4'), ('ay', 'f4'), ('az', 'f4')])
    with open(myfiles, "rb") as f:
        x = np.fromfile(f, dtype=dt)
        frequences = x['frequences'].astype('float32')
        vx = x['vx'].astype('float32')
    return frequences, vx
directory_path = "/home/noureddine/codes/efispec3d.git/efispec3d-interne/test/Dumanoir/Test-GSA-2"
num_simulations = 370
simulation_data1 = []
for i in range(1, num_simulations + 1):
    simu_name = f"SIMU{str(i).zfill(6)}"
    file_path = f"{directory_path}/{simu_name}/Test-GSA-2.fsr.000004.gpl.fft"
    frequences, vx = read_efi_binary_time_series_vx(file_path)
    simulation_data1.append((frequences, vx))
num_components = 2
num_temps_points = 16385
tensor_data_vx = np.zeros((num_simulations, num_components, num_temps_points))
for i, data1 in enumerate(simulation_data1):
    frequences = data1[0]
    vx = data1[1]
    tensor_data_vx[i, 0, :] = frequences
    tensor_data_vx[i, 1, :] = vx
frequences = tensor_data_vx[:, 0, :]
frequences= np.squeeze(frequences)
velocities_vx = tensor_data_vx[:, 1, :]
velocities_vx = np.squeeze(velocities_vx)
velocities_vx_max=np.max(velocities_vx, axis=1)
Uncertain_inputs_path_file= "/home/noureddine/codes/efispec3d.git/efispec3d-interne/test/Dumanoir/Test-une-couche-chachée/Test-GSA-2.uqc"
Uncertain_inputs= np.genfromtxt(Uncertain_inputs_path_file, delimiter=" ")
a_train=360
a_test=360

# Diviser les données entre train_set et test_set
Uncertain_inputs_train = Uncertain_inputs[:a_train]
X_train = Uncertain_inputs_train
y_train_vx = velocities_vx[:a_train]
y_min_vx = np.min(y_train_vx)
y_max_vx = np.max(y_train_vx)
y_train_normalized_vx = (y_train_vx - y_min_vx) / (y_max_vx - y_min_vx)
Uncertain_inputs_test = Uncertain_inputs[a_test:num_simulations]
x_test = Uncertain_inputs_test
y_test=velocities_vx[a_test:num_simulations]
y_test_normalised=(y_test - y_min_vx) / (y_max_vx - y_min_vx)

####
inputs_size=6
output_size=16385
hidden_neurones_number=250


###################Plan d'experience pour l'analyse de sensibilité globale en utilisant la suite de Sobol
from SALib.sample import saltelli
import numpy as np
nombre_lignes =4096
nombre_colonnes = 12
problem = {
    'num_vars': 12,
    'names': [f'input_{i}' for i in range(12)],
    'bounds': [[1150, 1650],[2150,2650],[3750,4250],[14068,15068],[17553.33,18583.33],[-28520,-27520],[1150, 1650],[2150,2650],[3750,4250],[14068,15068],[17553.33,18583.33],[-28520,-27520]]
}
matrice = saltelli.sample(problem, nombre_lignes)
######""Normaliser les entrées
min_colon1= np.min(matrice[:, 0])
max_colon1=np.max(matrice[:, 0])
X11 = (matrice[:, 0] - min_colon1) / (max_colon1 - min_colon1)
min_colon2= np.min(matrice[:, 1])
max_colon2=np.max(matrice[:, 1])
X21= (matrice[:, 1] - min_colon2) / (max_colon2- min_colon2)
min_colon3= np.min(matrice[:, 2])
max_colon3=np.max(matrice[:, 2])
X31= (matrice[:, 2] - min_colon3) / (max_colon3 - min_colon3)
min_colon4= np.min(matrice[:, 3])
max_colon4=np.max(matrice[:, 3])
X41 = (matrice[:, 3] - min_colon4) / (max_colon4 - min_colon4)
min_colon5= np.min(matrice[:, 4])
max_colon5=np.max(matrice[:, 4])
X51 = (matrice[:, 4] - min_colon5) / (max_colon5- min_colon5)
min_colon6= np.min(matrice[:, 5])
max_colon6=np.max(matrice[:, 5])
X61 = (matrice[:, 5] - min_colon6) / (max_colon6 - min_colon6)
min_colon7= np.min(matrice[:, 6])
max_colon7=np.max(matrice[:, 6])
X12 = (matrice[:, 6] - min_colon7) / (max_colon7 - min_colon7)
min_colon8= np.min(matrice[:, 7])
max_colon8=np.max(matrice[:, 7])
X22= (matrice[:, 7] - min_colon8) / (max_colon8- min_colon8)
min_colon9= np.min(matrice[:, 8])
max_colon9=np.max(matrice[:, 8])
X32= (matrice[:, 8] - min_colon9) / (max_colon9 - min_colon9)
min_colon10= np.min(matrice[:, 9])
max_colon10=np.max(matrice[:, 9])
X42 = (matrice[:, 9] - min_colon10) / (max_colon10- min_colon10)
min_colon11= np.min(matrice[:, 10])
max_colon11=np.max(matrice[:, 10])
X52= (matrice[:, 10] - min_colon11) / (max_colon11- min_colon11)
min_colon12= np.min(matrice[:, 11])
max_colon12=np.max(matrice[:, 11])
X62= (matrice[:, 11] - min_colon12) / (max_colon12 - min_colon12)


####Construction du modèle de réseaux de neurones de Liu
#Fonction d'activation
class CustomActivation_Liu(tf.keras.layers.Layer):
    def __init__(self, input_dim, **kwargs):
        super(CustomActivation_Liu, self).__init__(**kwargs)
        self.v0 = self.add_weight(shape=(1,), initializer='random_normal', trainable=True)
        self.vk = self.add_weight(shape=(input_dim,), initializer='random_normal', trainable=True)
        self.hk_cos = self.add_weight(shape=(input_dim,), initializer='random_normal', trainable=True)
        self.bk = self.add_weight(shape=(input_dim,), initializer='random_normal', trainable=True)
        self.uk = self.add_weight(shape=(input_dim,), initializer='random_normal', trainable=True)
        self.hk_sin = self.add_weight(shape=(input_dim,), initializer='random_normal', trainable=True)
        self.pk = self.add_weight(shape=(input_dim,), initializer='random_normal', trainable=True)
        self.qk = self.add_weight(shape=(input_dim,), initializer='random_normal', trainable=True)

    def call_Liu(self, inputs):
        result_Liu = self.v0
        for k in range(len(inputs)):
            result_Liu += self.vk[k] * tf.cos(self.hk_cos[k] * inputs + self.bk[k]) \
                      + self.uk[k] * tf.sin(self.hk_sin[k] * inputs + self.pk[k])
        return result_Liu

#Architecture
model_Liu = tf.keras.Sequential([
    tf.keras.layers.Dense(16, input_shape=(inputs_size,)),
    CustomActivation_Liu(inputs_size),
    tf.keras.layers.Dense(16),
    tf.keras.layers.Dense(16),
    tf.keras.layers.Dense(output_size, activation='linear')
])
# Compilation du modèle
learning_rate_Liu = 0.0045
optimizer_Liu= tf.keras.optimizers.Adam(learning_rate=learning_rate_Liu)
model_Liu.compile(optimizer=optimizer_Liu, loss='mse')
history_Liu=model_Liu.fit(X_train, y_train_normalized_vx, epochs=100, batch_size=2,validation_data=(x_test,y_test_normalised))
#Erreur d'entrainement et Erreur de généralisation
train_loss_Liu=history_Liu.history['loss']
test_loss_Liu=history_Liu.history['val_loss']
plt.plot(train_loss_Liu,label='Loss trainLiu')
plt.plot(test_loss_Liu,label='Loss testLiu')
plt.legend()
plt.show()

########## Modele finale de Liu
def model_final_Liu(input_1,input_2,input_3,input_4,input_5,input_6):
    inputs = np.column_stack((input_1, input_2,input_3,input_4,input_5,input_6))
    sortie_Liu=model_Liu.predict(inputs)
    reponse_Liu=sortie_Liu * (y_max_vx - y_min_vx) + y_min_vx
    return reponse_Liu
#########GSA
resultats_Liu= model_final_Liu(X11,X21,X31,X41,X51,X61)
moyenne_Liu=np.mean(resultats_Liu,axis=0)
moyenne_carré_Liu=moyenne_Liu**2
Variance_total_Liu=np.var(resultats_Liu,axis=0)
def indice_de_sobol_ordre1_Liu(input1,input2,input3,input4,input5,input6):
    reponse_echantillonconditionnee_Liu=model_final_Liu(input1,input2,input3,input4,input5,input6)
    a_Liu=resultats_Liu*reponse_echantillonconditionnee_Liu
    U_Liu=np.mean(a_Liu,axis=0)
    V_Liu=U_Liu-moyenne_carré_Liu
    return V_Liu/Variance_total_Liu
V_s1_Liu=indice_de_sobol_ordre1_Liu(X11,X22,X32,X42,X52,X62)
V_s2_Liu=indice_de_sobol_ordre1_Liu(X12,X21,X32,X42,X52,X62)
V_s3_Liu=indice_de_sobol_ordre1_Liu(X12,X22,X31,X42,X52,X62)
X_source_Liu=indice_de_sobol_ordre1_Liu(X12,X22,X32,X41,X52,X62)
Y_source_Liu=indice_de_sobol_ordre1_Liu(X12,X22,X32,X42,X51,X62)
Z_source_Liu=indice_de_sobol_ordre1_Liu(X12,X22,X32,X42,X52,X61)
indices_sobol_Liu = np.array([V_s1_Liu, V_s2_Liu, V_s3_Liu, X_source_Liu, Y_source_Liu, Z_source_Liu])
sobol_indices_matrix_Liu = np.array(indices_sobol_Liu)
cumulative_influence_Liu = np.zeros_like(sobol_indices_matrix_Liu[0])
noms_parametres = ['Vs1', 'Vs2', 'Vs3', 'Xsource', 'Ysource', 'Zsource']
n_params = sobol_indices_matrix_Liu.shape[0]
plt.figure(figsize=(10, 8))
for param in range(n_params):
    influence_Liu = sobol_indices_matrix_Liu[param]
    plt.fill_between(frequences[0], cumulative_influence_Liu, cumulative_influence_Liu + influence_Liu, alpha=0.3, label=noms_parametres[param])
    cumulative_influence_Liu += influence_Liu
plt.title('Méthode de Sobol')
plt.xlabel('Fréquence [Hz]')
plt.ylabel('Indices de Sobol cumulées')
plt.xlim(0, 1.25)
plt.legend()
plt.show()

####Construction du modèle de Silvesku
#Fonction d'activation
class CustomActivation_Silvesku(tf.keras.layers.Layer):
    def __init__(self, input_dim, **kwargs):
        super(CustomActivation_Silvesku, self).__init__(**kwargs)
        self.c = self.add_weight(shape=(input_dim,), initializer='random_normal', trainable=True)
        self.omega = self.add_weight(shape=(input_dim, input_dim), initializer='random_normal', trainable=True)
        self.phi = self.add_weight(shape=(input_dim, input_dim), initializer='random_normal', trainable=True)
    def call_Silvesku(self, inputs):
        resultats_Silvesku= tf.reduce_sum(self.c * tf.reduce_prod(tf.cos(self.omega * inputs + self.phi), axis=1), axis=1)
        return resultats_Silvesku
#Architecture
model_Silvescu = tf.keras.Sequential([
    tf.keras.layers.Dense(16, input_shape=(inputs_size,)),
    CustomActivation_Silvesku(inputs_size),
    tf.keras.layers.Dense(16),  # Couche cachée
    tf.keras.layers.Dense(output_size, activation='linear')
])
#Compilation du modèle
learning_rate_Silvesku = 0.0045
optimizer_Silvesku = tf.keras.optimizers.Adam(learning_rate=learning_rate_Silvesku)
model_Silvescu.compile(optimizer=optimizer_Silvesku, loss='mse')
history_Silvescu=model_Silvescu.fit(X_train, y_train_normalized_vx, epochs=100, batch_size=2,validation_data=(x_test,y_test_normalised))
train_loss_Silvescu=history_Silvescu.history['loss']
test_loss_Silvescu=history_Silvescu.history['val_loss']
#Erreur d'entrainement et Erreur de généralisation
plt.plot(train_loss_Silvescu,label='Loss trainSilvescu')
plt.plot(test_loss_Silvescu,label='Loss testSilvescu')
plt.legend()
plt.show()

#####Model final de Silvesku
def model_final_Silvesku(input_1,input_2,input_3,input_4,input_5,input_6):
    inputs = np.column_stack((input_1, input_2,input_3,input_4,input_5,input_6))
    sortie=model_Silvescu.predict(inputs)
    reponse_Silvesku=sortie * (y_max_vx - y_min_vx) + y_min_vx
    return reponse_Silvesku
############GSA
resultats_Silvesku= model_final_Silvesku(X11,X21,X31,X41,X51,X61)
moyenne_Silvesku=np.mean(resultats_Silvesku,axis=0)
moyenne_carré_Silvesku=moyenne_Silvesku**2
Variance_total_Silvesku=np.var(resultats_Silvesku,axis=0)
def indice_de_sobol_ordre1_Silvesku(input1,input2,input3,input4,input5,input6):
    reponse_echantillonconditionnee_Silvesku=model_final_Silvesku(input1,input2,input3,input4,input5,input6)
    a_Silvesku=resultats_Silvesku*reponse_echantillonconditionnee_Silvesku
    U_Silvesku=np.mean(a_Silvesku,axis=0)
    V_Silvesku=U_Silvesku-moyenne_carré_Silvesku
    return V_Silvesku/Variance_total_Silvesku
V_s1_Silvesku=indice_de_sobol_ordre1_Silvesku(X11,X22,X32,X42,X52,X62)
V_s2_Silvesku=indice_de_sobol_ordre1_Silvesku(X12,X21,X32,X42,X52,X62)
V_s3_Silvesku=indice_de_sobol_ordre1_Silvesku(X12,X22,X31,X42,X52,X62)
X_source_Silvesku=indice_de_sobol_ordre1_Silvesku(X12,X22,X32,X41,X52,X62)
Y_source_Silvesku=indice_de_sobol_ordre1_Silvesku(X12,X22,X32,X42,X51,X62)
Z_source_Silvesku=indice_de_sobol_ordre1_Silvesku(X12,X22,X32,X42,X52,X61)
indices_sobol_Silvesku = np.array([V_s1_Silvesku, V_s2_Silvesku, V_s3_Silvesku, X_source_Silvesku, Y_source_Silvesku, Z_source_Silvesku])
sobol_indices_matrix_Silvesku = np.array(indices_sobol_Silvesku)
cumulative_influence_Silvesku = np.zeros_like(sobol_indices_matrix_Silvesku[0])
noms_parametres = ['Vs1', 'Vs2', 'Vs3', 'Xsource', 'Ysource', 'Zsource']
n_params = sobol_indices_matrix_Silvesku.shape[0]
plt.figure(figsize=(10, 8))
for param in range(n_params):
    influence_Silvesku = sobol_indices_matrix_Silvesku[param]
    plt.fill_between(frequences[0], cumulative_influence_Silvesku, cumulative_influence_Silvesku + influence_Silvesku, alpha=0.3, label=noms_parametres[param])
    cumulative_influence_Silvesku += influence_Silvesku
plt.title('Méthode de Sobol')
plt.xlabel('Fréquence [Hz]')
plt.ylabel('Indices de Sobol cumulées')
plt.xlim(0,1.25)
plt.legend()
plt.show()

####Construction du modèle de réseau de neurones de Gallant and White
#Fonction d'activation
class CosineSquasher(tf.keras.layers.Layer):
    def __init__(self):
        super(CosineSquasher, self).__init__()
    def call(self, x):
        x_tf = tf.convert_to_tensor(x, dtype=tf.float32)  # Convertir x en un tenseur TensorFlow
        phi_x = 0.5 * (1 + tf.cos(x_tf + (3 * np.pi / 2)))
        indicator = tf.where(tf.logical_and(x_tf >= -np.pi / 2, x_tf <= np.pi / 2), 1.0, 0.0)  # Utiliser des flottants (1.0, 0.0) au lieu d'entiers (1, 0)
        return phi_x * indicator + tf.cast(x_tf > np.pi / 2, dtype=tf.float32)
#Architecture
model_GW= models.Sequential()
model_GW.add(layers.Dense(16, activation=CosineSquasher(), input_shape=(inputs_size,)))
model_GW.add(layers.Dense(output_size, activation='linear'))
#Compilation du modèle
learning_rate_GW = 0.0045
optimizer_GW= tf.keras.optimizers.Adam(learning_rate=learning_rate_GW)
model_GW.compile(optimizer=optimizer_GW, loss='mse')
history_GW=model_GW.fit(X_train, y_train_normalized_vx, epochs=100, batch_size=2,validation_data=(x_test,y_test_normalised))
#Erreur d'entrainement et Erreur de généralisation
train_loss_GW=history_GW.history['loss']
test_loss_GW=history_GW.history['val_loss']
plt.plot(train_loss_GW,label='Loss trainGW')
plt.plot(test_loss_GW,label='Loss testGW')
plt.legend()
plt.show()
####Modele final de Gallant and White
def model_final_GW(input_1,input_2,input_3,input_4,input_5,input_6):
    inputs = np.column_stack((input_1, input_2,input_3,input_4,input_5,input_6))
    sortie_GW=model_GW.predict(inputs)
    reponse_GW=sortie_GW * (y_max_vx - y_min_vx) + y_min_vx
    return reponse_GW
################GSA
resultats_GW= model_final_GW(X11,X21,X31,X41,X51,X61)
moyenne_GW=np.mean(resultats_GW,axis=0)
moyenne_carré_GW=moyenne_GW**2
Variance_total_GW=np.var(resultats_GW,axis=0)
def indice_de_sobol_ordre1_GW(input1,input2,input3,input4,input5,input6):
    reponse_echantillonconditionnee_GW=model_final_GW(input1,input2,input3,input4,input5,input6)
    a_GW=resultats_GW*reponse_echantillonconditionnee_GW
    U_GW=np.mean(a_GW,axis=0)
    V_GW=U_GW-moyenne_carré_GW
    return V_GW/Variance_total_GW
V_s1_GW=indice_de_sobol_ordre1_GW(X11,X22,X32,X42,X52,X62)
V_s2_GW=indice_de_sobol_ordre1_GW(X12,X21,X32,X42,X52,X62)
V_s3_GW=indice_de_sobol_ordre1_GW(X12,X22,X31,X42,X52,X62)
X_source_GW=indice_de_sobol_ordre1_GW(X12,X22,X32,X41,X52,X62)
Y_source_GW=indice_de_sobol_ordre1_GW(X12,X22,X32,X42,X51,X62)
Z_source_GW=indice_de_sobol_ordre1_GW(X12,X22,X32,X42,X52,X61)
indices_sobol_GW = np.array([V_s1_GW, V_s2_GW, V_s3_GW, X_source_GW, Y_source_GW, Z_source_GW])
sobol_indices_matrix_GW = np.array(indices_sobol_GW)
cumulative_influence_GW = np.zeros_like(sobol_indices_matrix_GW[0])
noms_parametres = ['Vs1', 'Vs2', 'Vs3', 'Xsource', 'Ysource', 'Zsource']
n_params = sobol_indices_matrix_GW.shape[0]
plt.figure(figsize=(10, 8))
for param in range(n_params):
    influence_GW = sobol_indices_matrix_GW[param]
    plt.fill_between(frequences[0], cumulative_influence_GW, cumulative_influence_GW + influence_GW, alpha=0.3, label=noms_parametres[param])
    cumulative_influence_GW += influence_GW
plt.title('Méthode Quasi-MC')
plt.xlabel('Fréquence [Hz]')
plt.ylabel('Indices de Sobol cumulées')
plt.xlim(0, 1.25)
plt.legend()
plt.show()

###############Modele de Réseau de neurones standard avec la fonction d'activation Relu
#Architecture
model_standard= models.Sequential()
model_standard.add(layers.Dense(hidden_neurones_number, activation="relu", input_shape=(inputs_size,)))
model_standard.add(layers.Dense(hidden_neurones_number, activation='relu'))
model_standard.add(layers.Dense(output_size, activation='linear'))
#Compilation du modèle
learning_rate_standard = 0.001
optimizer_standard = tf.keras.optimizers.Adam(learning_rate=learning_rate_standard)
model_standard.compile(optimizer=optimizer_standard, loss='mse')
history_standard=model_standard.fit(X_train, y_train_normalized_vx, epochs=100, batch_size=2,validation_data=(x_test,y_test_normalised))
train_loss_standard=history_standard.history['loss']
test_loss_standard=history_standard.history['val_loss']
##Erreur d'entrainement et Erreur de généralisation
plt.plot(train_loss_standard,label='Loss train')
plt.plot(test_loss_standard,label='Loss test')
plt.legend()
plt.show()
#############Modele finale du RN standard
def model_final_standard(input_1,input_2,input_3,input_4,input_5,input_6):
    inputs = np.column_stack((input_1, input_2,input_3,input_4,input_5,input_6))
    sortie_standard=model_standard.predict(inputs)
    reponse_standard=sortie_standard * (y_max_vx - y_min_vx) + y_min_vx
    return reponse_standard
#######GSA
resultats_standard= model_final_standard(X11,X21,X31,X41,X51,X61)
moyenne_standard=np.mean(resultats_standard,axis=0)
moyenne_carré_standard=moyenne_standard**2
Variance_total_standard=np.var(resultats_standard,axis=0)
def indice_de_sobol_ordre1_standard(input1,input2,input3,input4,input5,input6):
    reponse_echantillonconditionnee_standard=model_final_standard(input1,input2,input3,input4,input5,input6)
    a_standard=resultats_standard*reponse_echantillonconditionnee_standard
    U_standard=np.mean(a_standard,axis=0)
    V_standard=U_standard-moyenne_carré_standard
    return V_standard/Variance_total_standard
V_s1_Standard=indice_de_sobol_ordre1_standard(X11,X22,X32,X42,X52,X62)
V_s2_Standard=indice_de_sobol_ordre1_standard(X12,X21,X32,X42,X52,X62)
V_s3_Standard=indice_de_sobol_ordre1_standard(X12,X22,X31,X42,X52,X62)
X_source_Standard=indice_de_sobol_ordre1_standard(X12,X22,X32,X41,X52,X62)
Y_source_Standard=indice_de_sobol_ordre1_standard(X12,X22,X32,X42,X51,X62)
Z_source_Standard=indice_de_sobol_ordre1_standard(X12,X22,X32,X42,X52,X61)
indices_sobol_Standard = np.array([V_s1_Standard, V_s2_Standard, V_s3_Standard, X_source_Standard, Y_source_Standard, Z_source_Standard])
sobol_indices_matrix_standard = np.array(indices_sobol_Standard)
cumulative_influence_standard = np.zeros_like(sobol_indices_matrix_standard[0])
noms_parametres = ['Vs1', 'Vs2', 'Vs3', 'Xsource', 'Ysource', 'Zsource']
n_params = sobol_indices_matrix_standard.shape[0]
plt.figure(figsize=(10, 8))
for param in range(n_params):
    influence_standard = sobol_indices_matrix_standard[param]
    plt.fill_between(frequences[0], cumulative_influence_standard, cumulative_influence_standard + influence_standard, alpha=0.3, label=noms_parametres[param])
    cumulative_influence_standard += influence_standard
plt.title('Méthode Quasi-MC')
plt.xlabel('Fréquence [Hz]')
plt.ylabel('Indices de Sobol cumulées')
plt.xlim(0, 1.25)
plt.legend()
plt.show()

######Loss pour chaque model
plt.plot(test_loss_standard,label='Loss test Standard')
plt.plot(test_loss_GW,label='Loss test GW ')
plt.plot(test_loss_Silvescu,label='Loss test Silvescu')
plt.plot(test_loss_Liu,label='Loss test Liu ')
plt.ylim(4e-07,7e-05)
plt.legend()
plt.show()

###########################comparaison entre les différentes modèles et le modele EFISPEC3D sur l'ensemble de test
velocities_Liu_predi=model_Liu.predict(x_test)
velocities_Liu_predi_denormalised=velocities_Liu_predi* (y_max_vx - y_min_vx) + y_min_vx
velocities_Silvesku_predi=model_Silvescu.predict(x_test)
velocities_Silvesku_predi_denormalised=velocities_Silvesku_predi* (y_max_vx - y_min_vx) + y_min_vx
velocities_GW_predi=model_GW.predict(x_test)
velocities_GW_predi_denormalised=velocities_GW_predi* (y_max_vx- y_min_vx) + y_min_vx
velocities_Standard_predi=model_standard.predict(x_test)
velocities_Standard_predi_denormalised=velocities_Standard_predi* (y_max_vx - y_min_vx) + y_min_vx
for i in range(10):
    plt.figure()
    plt.plot(frequences[i], velocities_Standard_predi_denormalised[i], linewidth=0.7, label=f'Standard-NN')
    plt.plot(frequences[i], velocities_Liu_predi_denormalised[i], linewidth=0.7, label=f'Liu-NN')
    plt.plot(frequences[i], velocities_Silvesku_predi_denormalised[i], linewidth=0.7, label=f'Silvesku-NN')
    plt.plot(frequences[i], velocities_GW_predi_denormalised[i], linewidth=0.7, label=f'GW-NN')
    plt.plot(frequences[i], velocities_vx[a_test+i],label='SEM')
    plt.xlabel('Frequency [Hz]')
    plt.xlim(0,1.25)
    plt.ylabel('EW-velocity')
    plt.legend()
    plt.show()

#######Indice de Sobol pour le V_s1 selon chaque modèle
plt.figure(figsize=(10,6))
plt.plot(frequences[0],V_s1_GW, linewidth=0.7, label=f'Standard-NN')
plt.plot(frequences[0],V_s1_Liu, linewidth=0.7, label=f'Liu-NN')
plt.plot(frequences[0],V_s1_Silvesku, linewidth=0.7, label=f'Silvesku-NN')
plt.plot(frequences[0],V_s1_Standard, linewidth=0.7, label=f'GW-NN')
plt.xlabel('Frequency [Hz]')
plt.xlim(0,1.25)
plt.ylabel('Indices de sobol')
plt.legend()
plt.show()