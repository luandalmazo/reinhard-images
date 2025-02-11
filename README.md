##README

Mudanças feitas para a execução do experimento previamente proposto na thanos (VM do dinf).
Falta polir alguns imports e funções, tais como a biblioteca de environments do python.
Fiz um script que executa o treino e teste dos dados com a normalização de reinhard ou normais. Se preferível, pode-se mudar para que seja tudo executado em uma só vez.
Alguns outros detalhes que ainda podem ser melhorados:
- Uso da classe proposta mais consistentemente e calcular a média e o std dos dados em tempo de exeucção, sem precisar mudar o código para preencher o cálculo manualmente
- Aplicar a normalização de reinhard aos dados como parte do script, de forma que a entrada seja apenas os dados 'normais', e a partir disso o experimento seja inteiramente feito
- O script 'testing_normalization.py' está para a execução dos dados 'normais' como padrão, e as 20 épocas também. Uma parte dos dados previamente salvos em ambientes foram transformados em parâmetros.
