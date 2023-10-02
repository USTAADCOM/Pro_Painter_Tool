# download aot-ckpt 
wget -P ./ckpt https://github.com/USTAADCOM/Pro_Painter_Tool/releases/download/v1.1/R50_DeAOTL_PRE_YTB_DAV.pth
# download sam-ckpt
wget -P ./ckpt https://github.com/USTAADCOM/Pro_Painter_Tool/releases/download/v1.1/sam_vit_b_01ec64.pth
# download grounding-dino ckpt
wget -P ./ckpt https://github.com/USTAADCOM/Pro_Painter_Tool/releases/download/v1.1/groundingdino_swint_ogc.pth
# download Pro Painter Models
wget -P ./ProPainter/weights https://github.com/USTAADCOM/Pro_Painter_Tool/releases/download/v1.1/i3d_rgb_imagenet.pt
wget -P ./ProPainter/weights https://github.com/USTAADCOM/Pro_Painter_Tool/releases/download/v1.1/ProPainter.pth
wget -P ./ProPainter/weights https://github.com/USTAADCOM/Pro_Painter_Tool/releases/download/v1.1/raft-things.pth
wget -P ./ProPainter/weights https://github.com/USTAADCOM/Pro_Painter_Tool/releases/download/v1.1/recurrent_flow_completion.pth