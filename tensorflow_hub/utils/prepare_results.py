from utils import translate as tr

def sort_translate_print(result_list):
  i = 1
  sorted_values = sorted(result_list, reverse=True, key=result_list.__getitem__)
  request_body = "text="
  lang = "en-lv"
  
  for s in sorted_values:
      request_body += "{}.".format(s)
  request_body = request_body[:-1]
  response = tr.translate(lang, request_body)
  
  print("=======================================")
  if (response["code"] == 200):
      response_body = response['text'][0]
      translation_results = response_body[:-1].split(".")
      for k in sorted_values:
        print("[{}] {} % - {} ({})".format(i, result_list[k], translation_results[i-1], k))
        i += 1
  else:
     for k in sorted_values:
        print("[{}] {} % - {} ({})".format(i, result_list[k], k))
        i += 1
  print("=======================================")