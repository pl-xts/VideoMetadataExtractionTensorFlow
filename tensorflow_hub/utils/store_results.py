

def store_results(name, num_of_classes, avg_score, time, video_type):
    path = './results_' + video_type + '.txt'
    file = open(path,'a+')
    string = "{};{};{};{};{}\n".format(name, num_of_classes, round(avg_score, 2), time, video_type)
    file.write(string)
    file.close()
# store_results('resnet', 100, 55, "155", "night")
# file.close()
# reader = open(path, 'r')
# for f in reader.readlines():
#   print(f.strip())