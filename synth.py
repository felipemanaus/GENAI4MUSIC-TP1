import os
import subprocess

arquivo_midi = 'transicao_daft_punk.mid'
arquivo_soundfont = 'Vintage Dreams Waves v2.sf2'
arquivo_saida = 'track_final.wav'

if not os.path.exists(arquivo_soundfont):
    print(f"ERRO: O arquivo '{arquivo_soundfont}' não foi encontrado!")
elif not os.path.exists(arquivo_midi):
    print(f"ERRO: O arquivo '{arquivo_midi}' não foi encontrado!")
else:
    print("Arquivos encontrados! Executando FluidSynth com a sintaxe atualizada...")
    
    comando = [
        'fluidsynth',
        '-ni', 
        '-F', arquivo_saida,  
        '-r', '44100',        
        arquivo_soundfont,    
        arquivo_midi          
    ]
    
    subprocess.run(comando)
    
    if os.path.exists(arquivo_saida):
        print(f"\nSUCESSO! O arquivo '{arquivo_saida}' foi gerado perfeitamente.")
    else:
        print("\nA conversão falhou.")