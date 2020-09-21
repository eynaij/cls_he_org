import json
import pandas as pd
import glob
import random
from tqdm import tqdm

json_dir_path = "/data/hejy/hejy_dp/datasets/Adobe_Synthetic_Dataset/JSONs"
csv_dir_path = "//data/hejy/hejy_dp/datasets/Adobe_Synthetic_Dataset/"

image_id_list = []


# area_list = []; heatmap_list = [];horizontal_bar_list = [];horizontal_interval_list = [];line_list = [];manhattan_list = [];map_list = []
# pie_list = [];scatter_list = [];scatter_line_list = [];surface_list = [];venn_list = [];vertical_bar_list = [];vertical_box_list = []
# vertical_interval_list = []

# type_list = ['area', 'heatmap','horizontal_bar', 'horizontal_interval', 'line', 'manhattan','map','pie',\
#             'scatter', 'scatter-line','surface','venn','vertical_bar','vertical_box','vertical_interval']
# type_list_list = [area_list, heatmap_list, horizontal_bar_list, horizontal_interval_list, line_list, manhattan_list, map_list, \
#                  pie_list, scatter_list,scatter_line_list, surface_list, venn_list, vertical_bar_list,vertical_box_list, vertical_interval_list]

area_list = []; donut_list = []; hbox_list = []; hGroup_list = [];hStack_list = [];line_list = [];pie_list = [];polar_list = [];scatter_list = [];vbox_list=[];vGroup_list=[];vStack_list=[]
type_list = ['area', 'donut','hbox', 'hGroup','hStack', 'line', 'pie', 'polar', 'scatter', 'vbox', 'vGroup', 'vStack']
type_map_dict={'Area':'area', 'Donut':'donut','Horizontal box':'hbox', "Grouped horizontal bar":'hGroup', "Stacked horizontal bar":'hStack', "Line":'line',\
     "Pie":'pie', "Polar":'polar', "Scatter":'scatter',  "Vertical box":'vbox', "Grouped vertical bar":'vGroup', "Stacked vertical bar":'vStack'}
type_list_list = [area_list, donut_list, hbox_list, hGroup_list, hStack_list, line_list, pie_list, polar_list, scatter_list, vbox_list, vGroup_list, vStack_list]

file_list = glob.glob(json_dir_path + '/*/*.json')
random.shuffle(file_list)
for i, json_file_path in tqdm(enumerate(file_list)):
    json_file = json.load(open(json_file_path,'rb'))
    chart_type = json_file['task1']['output']['chart_type']
    # if ' ' in chart_type:
    #     chart_type = '_'.join(chart_type.split(' '))
    if not chart_type:
        print("chart_type of %s is None" %json_file_path)
    for _ in type_list_list:
        _.append(0)
    type_list_list[type_list.index(type_map_dict[chart_type])][i] = 1
    image_id = type_map_dict[chart_type] + '/'+json_file_path.split('/')[-1].rstrip('.json')
    image_id_list.append(image_id)

type_dict = {}
type_dict['image_id'] = image_id_list
for key, value in zip(type_list, type_list_list):
    type_dict[key] = value
dataframe = pd.DataFrame(type_dict)
dataframe.to_csv(csv_dir_path+'/synthetic_train.csv', index=False,sep=',' )
