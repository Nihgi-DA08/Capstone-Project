# CAPSTONE PROJECT
Phân tích về lượng booking khách sạn của Bồ Đào Nha năm 2015-2017.

## HÌNH ẢNH DEMO
<p align="center">
<img src='pic/0.png'></img>
</p>

## CODE DEMO
```python
# Create dropdown list
def crt_ddl(col_name, desc, percent='15%', default='All'):
    list = df[col_name].unique().tolist()
    list.append('All')
    return Dropdown(
        options=list,
        value=default,
        description=f'{desc}: ',
        style={'description_width': 'initial'},
        layout={'width': percent, 'margin': '0 auto'})
```

### THÀNH VIÊN
Nhóm NIHGI gồm các thành viên:

<img src='pic/1.jpg' align='right' width='16%' height='16%'></img>
<div style='display:flex;'>

- Nguyễn Đặng Trường An
- Nguyễn Hồng Phương Nghi
- Trần Văn Ninh (đã rời nhóm)
- Nguyễn Thu Trang
- Lương Thụy Vi (đã rời nhóm)

</div>

### TÍCH HỢP
<img src='pic/2.png' align='left' width='3%' height='3%'></img>
<div style='display:flex;'>

- Matplotlib » 3.5.2

</div>
<img src='pic/3.png' align='left' width='3%' height='3%'></img>
<div style='display:flex;'>

- NumPy » 1.21.5

</div>
<img src='pic/4.png' align='left' width='3%' height='3%'></img>
<div style='display:flex;'>

- Pandas » 1.4.4

</div>
<img src='pic/5.png' align='left' width='3%' height='3%'></img>
<div style='display:flex;'>

- seaborn » 0.11.2

</div>
<img src='pic/6.png' align='left' width='3%' height='3%'></img>
<div style='display:flex;'>

- Plotly » 2.9.3

</div>
<img src='pic/7.png' align='left' width='3%' height='3%'></img>
<div style='display:flex;'>

- scikit-learn » 1.0.2

</div>
