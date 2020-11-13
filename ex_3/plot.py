import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("./result.csv")

# Plots
## CPU
cpu = data[data["gpu_or_cpu"] == "cpu"]
cpu = cpu.groupby(["num_particles"]).first()
cpu = cpu[["execution_time"]]

cpu.plot(style=".", logx=True, logy=True)
plt.xlabel("Number of particles")
plt.ylabel("Execution time (s)")
plt.show()

## GPU
gpu = data[data["gpu_or_cpu"] == "gpu"]
data_gpu = pd.DataFrame()
for block_size, tmp_gpu in gpu.groupby(["block_size"]):
    tmp_gpu = tmp_gpu.set_index("num_particles")[["execution_time"]]
    tmp_gpu = tmp_gpu.rename(columns={"execution_time": "Block size = " + str(block_size)})
    data_gpu = pd.concat([data_gpu, tmp_gpu], axis=1)

axes = data_gpu.plot(subplots=True, layout=(2,3), style=".", logx=True, logy=True)
for ax in axes.flatten():
    ax.set_xlabel('Number of particles')
    ax.set_ylabel("Execution time (s)")
plt.show()

# Format for latex table

pd.set_option('display.max_colwidth', 5000)
## CPU
data_cpu = cpu.round(6)
data_cpu["latex"] = data_cpu.index.to_series().apply(str) + " & " + data_cpu["execution_time"].apply(str) + "\\\\"
print(data_cpu["latex"].to_string(index=False))

## GPU
data_gpu = data_gpu.round(6)
data_gpu["latex"] = data_gpu.index.to_series().apply(str) + " & " + data_gpu.iloc[:,0].apply(str) + " & " + data_gpu.iloc[:,1].apply(str) + " & " + data_gpu.iloc[:,2].apply(str) + " & " + data_gpu.iloc[:,3].apply(str) + " & " + data_gpu.iloc[:,4].apply(str) + "\\\\"

print(data_gpu["latex"].to_string(index=False))

