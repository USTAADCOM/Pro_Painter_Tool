# Pro_Painter_Tool
 Pro_Painter_Tool is a tool which take a video and text prompt as input. And remove the prompt object from the video.

## Setup
  ```code
  conda create -n <env_name> python==3.10
  conda activate <env_name>
  git clone https://github.com/USTAADCOM/Pro_Painter_Tool.git
  cd Pro_Painter_Tool
  pip install -r requirements.txt -q
  ```
## Install dependencied
  ```code
  bash script/install.sh
  ```
## Download Models
  ```code
  mkdir ./ckpt
  bash script/download_ckpt.sh
```
## Run Gradio Demo
```code
python3 app.py 
```
## Results

#### 👨🏻‍🎨 Object Removal
<table>
<tr>
   <td> 
      <img src="assets/bike_input.gif">
   </td>
    <td> 
      <img src="assets/bike_mask.gif">
   </td>
    <td> 
      <img src="assets/bike_output.gif">
   </td>
</tr>
</table>