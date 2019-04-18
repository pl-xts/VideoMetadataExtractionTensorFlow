from utils import translate as tr
import numpy as np
import matplotlib.pyplot as plt

def create_bar_chart(names, scores, model_name):
  y_pos = np.arange(len(names))
 
  plt.barh(y_pos, scores, align='center', alpha=0.5)
  plt.yticks(y_pos, tuple(names))
  plt.xlabel('Precizitāte')
  plt.ylabel('Klases')
  plt.title(model_name)
  plt.show()
def create_request_and_translate(sorted_names):
  
  request_body = "text="
  for s in sorted_names:
      request_body += "{}.".format(s)
  request_body = request_body[:-1]

  lang = "en-lv"
  response = tr.translate(lang, request_body)
  return response

def sort_translate_print(result_list, model_name):
 
  sorted_names = sorted(result_list, reverse=True, key=result_list.__getitem__)

  response = create_request_and_translate(sorted_names)
  
  i = 1
  print("=======================================")
  if (response["code"] == 200):
      response_body = response['text'][0]
      translation_results = response_body[:-1].split(".")
      scores = []
      names = []
      for k in sorted_names:
        print("[{}] {} % - {} ({})".format(i, result_list[k], translation_results[i-1], k))
        scores.append(result_list[k])
        names.append("{} ({})".format(translation_results[i-1], k))
        i += 1
  else:
     for k in sorted_names:
        print("[{}] {} % - {} ({})".format(i, result_list[k], k))
        scores.append(result_list[k])
        names.append("{}".format(k))
        i += 1
  print("=======================================")

  create_bar_chart(names, scores, model_name)
