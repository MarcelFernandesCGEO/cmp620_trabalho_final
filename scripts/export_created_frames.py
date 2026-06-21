import os
from qgis.core import (QgsProject, QgsVectorFileWriter, 
                       QgsCoordinateReferenceSystem, QgsVectorLayer, 
                       QgsCoordinateTransform, QgsWkbTypes)

# 1. Configurações de caminhos e nomes
camada_nome = "Created Frames"
pasta_saida = r"C:\Users\marcel.1CGEO\Desktop\GitHub\cmp620\trabalhos\trabalho_final\dados\bbox"
campo_nome = "mi"
epsg_alvo = "EPSG:4326"

# 2. Pegar a camada do projeto
camadas = QgsProject.instance().mapLayersByName(camada_nome)

if not camadas:
    print(f"Erro: Camada '{camada_nome}' não encontrada no projeto.")
else:
    camada = camadas[0]
    crs_alvo = QgsCoordinateReferenceSystem(epsg_alvo)
    contexto_transformacao = QgsProject.instance().transformContext()
    
    # Garantir que a pasta de saída existe
    if not os.path.exists(pasta_saida):
        os.makedirs(pasta_saida)

    # Pegar o tipo de geometria correto (ex: 'Polygon', 'MultiPolygon')
    tipo_geometria = QgsWkbTypes.displayString(camada.wkbType())

    # 3. Iterar sobre as feições
    contador = 0
    for feature in camada.getFeatures():
        # Obter o valor do atributo 'mi' e fazer o replace
        valor_mi = str(feature[campo_nome])
        nome_arquivo = valor_mi.replace("-", "_") + ".gpkg"
        caminho_completo = os.path.join(pasta_saida, nome_arquivo)
        
        # Criar a camada temporária usando a string de geometria limpa
        camada_temp = QgsVectorLayer(f"{tipo_geometria}?crs={camada.crs().authid()}", "temp", "memory")
        provedor_temp = camada_temp.dataProvider()
        
        # Copiar a estrutura de campos e adicionar a feição
        provedor_temp.addAttributes(camada.fields())
        camada_temp.updateFields()
        provedor_temp.addFeatures([feature])
        
        # Configurar as opções de salvamento e reprojeção
        opcoes_salvamento = QgsVectorFileWriter.SaveVectorOptions()
        opcoes_salvamento.driverName = "GPKG"
        opcoes_salvamento.ct = QgsCoordinateTransform(camada_temp.crs(), crs_alvo, QgsProject.instance())
        
        # 4. Exportar no padrão QGIS 4 (Apenas uma variável de retorno)
        resultado = QgsVectorFileWriter.writeAsVectorFormatV3(
            camada_temp,
            caminho_completo,
            contexto_transformacao,
            opcoes_salvamento
        )
        
        # No QGIS 4, writeAsVectorFormatV3 retorna 0 (QgsVectorFileWriter.WriterError.NoError) se der certo
        if resultado == QgsVectorFileWriter.WriterError.NoError:
            print(f"Sucesso: {nome_arquivo} exportado.")
            contador += 1
        else:
            print(f"Erro ao exportar {nome_arquivo}. Código do erro: {resultado}")

    print(f"\n--- Processo finalizado! {contador} feições exportadas. ---")