experiment_to_sync = "sis"
sizes_to_sync = ["100"]

origin_host = "cthulhu"
origin_dir = "/home/DATA/datasets/TEP/"
target_dir = "/home/tim/Documents/overleaf/TEP/"
result_dir = "gillespie_SIS/results/"

# get the avaliaable dirs in the origin
cmd = `ssh $(origin_host) -t "ls $(origin_dir)$(result_dir)$(experiment_to_sync)"`
output = read(cmd, String)
networks = split(output, "\n")

for size in sizes_to_sync
    for network in networks
        mkpath("$(target_dir)$(result_dir)$(experiment_to_sync)/$(network)/N$(size)")
        cmd = `rsync -avP --exclude 'mim*' "$(origin_host):$(origin_dir)$(result_dir)$(experiment_to_sync)/$(network)/N$(size)/*" "$(target_dir)$(result_dir)$(experiment_to_sync)/$(network)/N$(size)/"`
        run(cmd)
    end
end
