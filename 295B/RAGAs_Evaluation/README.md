# to disable the need of SUDO everytimt
```bash
```


# to view VScode on browser via SSH tunneling
## Local
```bash
$ ⁠ ssh -L 8000:127.0.0.1:8000 <student_id>@10.31.96.168 ⁠
$ ⁠ ssh -L 8000:127.0.0.1:8000 016649880@10.31.96.168 ⁠# e.g
```

## Cloud:
### run the VS Code server 
```bash
$ ⁠ code serve-web --accept-server-license-terms ⁠
$ code --verbose serve-web --accept-server-license-terms
```

# to view STREAMLIT on browser via SSH tunneling
## Local
```bash
ssh -L 8501:localhost:8501 016649880@10.31.96.168
```

## Cloud:
### run Streamlit
```bash
$ ⁠ streamlit run app1.py
```




