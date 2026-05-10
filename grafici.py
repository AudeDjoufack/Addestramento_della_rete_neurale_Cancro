import csv
import matplotlib.pyplot as plt

x_vals, y_vals = [], [] # queste variabili conterranno i dati contenuti nel file csv
# deve essere aggiornato ad ogni grafico per non accumulare dati corrispondenti a grafici diversi

# Leggiamo dal CSV
with open("Dati_Training_loss.csv", mode="r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        x_vals.append(float(row["epoch"]))
        y_vals.append(float(row["loss"]))

# Creiamo il grafico
plt.plot(x_vals, y_vals, marker="o")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("Grafico da dati di training")
plt.grid(True)
plt.show()

x_vals, y_vals = [], []

with open("Dati_Test_loss.csv", mode="r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        x_vals.append(float(row["epoch"]))
        y_vals.append(float(row["loss"]))

# Creiamo il grafico
plt.plot(x_vals, y_vals, marker="o")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("Grafico da dati di test")
plt.grid(True)
plt.show()

x_vals, y_vals = [], []

with open("Dati_Training_total_loss.csv", mode="r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        x_vals.append(float(row["epoch"]))
        y_vals.append(float(row["loss"]))

# Creiamo il grafico
plt.plot(x_vals, y_vals, marker="o")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("Training: la loss function totale di ogni epoch")
plt.grid(True)
plt.show()

x_vals, y_vals = [], []

with open("Dati_Test_total_loss.csv", mode="r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        x_vals.append(float(row["epoch"]))
        y_vals.append(float(row["loss"]))

# Creiamo il grafico
plt.plot(x_vals, y_vals, marker="o")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("Test: la loss function totale di ogni epoch")
plt.grid(True)
plt.show()

x_vals, y_vals = [], []

with open("Dati_Training_accuracy.csv", mode="r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        x_vals.append(float(row["epoch"]))
        y_vals.append(float(row["loss"]))

# Creiamo il grafico
plt.plot(x_vals, y_vals, marker="o")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("training: l'accuracy di ogni epoch")
plt.grid(True)
plt.show()

x_vals, y_vals = [], []

with open("Dati_Test_accuracy.csv", mode="r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        x_vals.append(float(row["epoch"]))
        y_vals.append(float(row["loss"]))

# Creiamo il grafico
plt.plot(x_vals, y_vals, marker="o")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("test: l'accuracy di ogni epoch")
plt.grid(True)
plt.show()

