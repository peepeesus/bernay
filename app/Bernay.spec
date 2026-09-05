# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for the Bernay desktop shell.
# pywebview ships JS/native assets that PyInstaller cannot see; locate them
# from the INSTALLED package rather than a hardcoded venv path.
import os
import webview
from PyInstaller.utils.hooks import collect_all

_wv = os.path.dirname(webview.__file__)
datas = [(os.path.join(_wv, 'js'), os.path.join('webview', 'js')),
         (os.path.join(_wv, 'lib'), os.path.join('webview', 'lib'))]
binaries = []
hiddenimports = ['webview.platforms.winforms', 'webview.platforms.edgechromium', 'clr']
tmp_ret = collect_all('pythonnet')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['desktop.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='Bernay',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['bernay.ico'],
)
