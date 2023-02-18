# fronteira_decisao_simulada

Só funciona para classificação de uma classe e fronteira de decisão linear.  Nos parâmetros dá pra alterar um pouco a distribuição dos dados, e no meu computador vai bem até uns 20000 exemplos.

Dá pra mexer na distribuição de dados através da variável expoentes. Exemplo: expoentes = (4, 0.1)
Se os exemplos de alvo fora da decisão estiverem em lugares ruins, dá pra corrigir isso pela variável ruido (numérico). Quanto mais alto o valor, mais espalhados vão ficar os falsos negativos.