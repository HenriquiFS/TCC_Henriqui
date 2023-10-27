from functions import *

positive_comments = [
    'eu adoro esse filme',
    'eu amo esse filme',
    'eu gosto desse filme',
    'gostei do filme, recomendo assistir, vale a pena',
    'gostei muito do filme, vou assistir de novo',
    'esse filme é bem melhor do que o último',
    'eu votei nele e vou votar de noto',
    'eu confio nele é um cara honesto',
    'gosto de morar aqui, não pretendo me mudar',
    'essa música é muito boa, mas não é a única boa deles, eles tem várias músicas legais',
    'é um carro economico, de fácil manutanção e por um bom preço',
    'comprei esse carro a 10 anos e ainda continuo com ele, não pretendo vender',
    'eu quero muito comprar essa moto, ela é meu sonho de consumo',
    'preço baixo mas grande retorno, vale a pena comprar',
    'encontrei com ela uma vez e gostei muito de conversar com ela',
    'valeu cara, me ajudou muito',
    'kkkkkkk não acredito nisso, é bom de mais pra ser verdade',
    'só fui uma vez lá, mas preciso voltar porque é muito legal',
    'esse pré treino dá muita energia, depois que você começa a tomar ele, você nem pensa mais em treinar sem ele',
    'já fui treinar nessa academia, é espaçosa e os equipamentos são bons, o preço é um pouco alto, mas os professores são atenciosos',
    'finalmente aquele calorão acabou, esse friozinho é bom de mais',
    'adotar um gato foi uma das melhores coisas que eu fiz, queria ter adotado 2 aquele dia',
    'esse canal é muito bom, ele vai direto ao ponto e explica muito bem',
    'foi a matéria que eu mais gostei na faculdade, mesmo todo mundo falando que era difícil eu fui bem',
    'isso é bom de mais pra ser verdade, parece até mentira, mas não é',
    'nunca vi um algoritmo tão bom quanto esse',
    'eles são meio malucos, mas fazem umas músicas muito boas',
    'esse aplicatívo é super útil para quem esta começando agora',
    'nunca vi um aplicativo ficar tão famoso assim tão rápido',
    'caramba, finalmente uma boa notícia, fiquei até até feliz em ouvir isso',
    'kkkkkkkk esse cara é muito louco, por isso que eu gosto dele',
    'nunca vi uma mulher tão bonita e empenhada como ela',
    'parabéns, você merece todo esse sucesso pelo seu empenho e dedicação',
    'no começo parece difícil, mas depois que você pega o jeito fica muito legal',
    'ufa, ainda bem que deu certo',
    'eu achei muito bom mesmo, me surpreendeu',
    'pra quem começou a treinar a menos de 1 ano, você esta indo muito bem mesmo',
    'Goodfellas é um dos meus filmes favoritos',
    'Pulp Fiction é um dos meus filmes favoritos, já assisti várias vezes',
    'depois que eu comecei a tomar pré treino, eu tive muito mais energia',
    'é um pouco caro, mas vale a pena pois aumenta muito o seu desempenho',
    'esse carro pode até ser caro, mas é tão lindo que vale a pena gastar o dinheiro',
    'eu ri muito quando assisti esse filme, é bem engraçado kkkkk',
    'essa cor realmente combina de mais com esse carro, ficou muito bonito',
    'pode até ser difícil de aprender, mas é algo que adiciona muito valor ao seu currículo',
    'essa chuva chegou na hora certa',
    'essa promoção esta muito boa, vou comprar antes que acabe',
    'mesmo com a chuva, valeu a pena ter ido, me diverti bastante',
    'trabalhei por 5 anos lá e realmente valeu a pena, aprendi bastante coisa',
    'depois de passar uns 2 anos guardando dinheiro, eu finalmente consegui comprar um, valeu a pena'
]

negative_comments = [
    'eu não gosto desse filme',
    'eu odeio esse filme',
    'esse filme é chato',
    'fui assistir o filme, mas achei chato, não recomendo',
    'já fui treinar nessa academia, mas achei muito cheia e os professores nem dão atenção aos alunos',
    'essa moto bebe de mais e a manutenção é cara',
    'esse carro é feio e não tem potencia nenhuma, mesmo sendo barato não vale a pena',
    'quem gosta de todo esse calor só pode ser louco',
    'quem gosta de todo esse frio só pode ser louco',
    'nem vejo a hora desse calor acabar e o inverno chegar',
    'eu gosto é de frio, esse calor me faz querer ir morar em outro país',
    'nunca vi um calor tão forte quanto esse, e o pior é que ainda estamos no fim do inverno',
    'assisti uns vídeos desse canal, mas achei eles muito superficiais e sem profundidade, não recomendo',
    'foi a matéria mais difícil que eu tive na faculdade, peguei DP 2 vezes',
    'já vai tarde, mereceu perder o cargo',
    'esse algoritmo leva muito tempo para rodar e tem baixa precisão',
    'quem acha que a situação vai melhorar só pode estar delirando',
    'participei só uma vez do evento, mas achei tão chato, se for igual ao último eu não vou ir nesse',
    'quem confia nesse cara deve ser muito ingênuo mesmo',
    'meu Deus, deviam banir logo esse cara',
    'bem feito, ele mereceu ser banido, ficava enchendo o saco de todo mundo',
    'nunca mais fui pra lá, achei o lugar muito feio e sem graça',
    'caramba, quem é burro o suficiente pra comprar esse carro?',
    'quem fala esse tipo de coisa deve ser muito inseguro',
    'kkkkkk bem feito, eles não mereciam ganhar mesmo',
    'só faria isso de novo se fosse obrigado a fazer',
    'já trabalhei lá, é uma empresa que só tem nome, mas na realidade é bem medíocre',
    'eu nunca disse isso, se alguém te falou isso, essa pessoa que era o mentiroso',
    'caramba, como que ainda tem gente que acredita nisso?',
    'só compra esse carro quem quer se mostrar, porque na real ele nem é potente',
    'meu maior arrependimento foi ter comprado essa moto, não faça isso',
    'ele só pode estar louco se esta achando que isso é uma boa ideia',
    'eu sabia que era bom de mais pra ser verdade, tava na cara que era uma mentira',
    'meu Deus, mas que tempo maluco é esse??? só faz calor agora nessa época',
    'eu não sabia que foi ele que fez isso, por isso que ninguém mais confia nele',
    'ele enganou muitas pessoas, por isso merece pagar pelo o que fez',
    'se ele tivesse se dedicado mais, não estaria onde esta hoje',
    'minhas espectativas não eram boas, mas isso é pior do que eu imaginava kkkkk',
    'o cara treina a mais de 5 anos mas ainda esta desse jeito kkkkk',
    'caramba, ele tinha tudo pra dar certo, mas jogou todas as chances no lixo',
    'esse cara é muito incompetente pra deixar um erro tão grotesco como esse acontecer',
    'a cada dia a corrupção só aumenta e ninguém faz nada a respeito',
    'eu estava tão empolgado para assistir esse filme, mas quando cheguei no cinema eu me decepcionei',
    'fiquei 1 hora na fila só pra comprar o ingresso e depois eles cancelaram o show',
    'essa promoção é um mentira, o preço sempre foi esse',
    'queria ter ido, mas estava muito cansado',
    'eu devia ter ficado em casa, estava um Sol forte, muito quente e cheio de gente',
    'a passagem está tão cara que nem compensa ir pra lá',
    'eu queria ter ido, mas não consegui tirar folga do trabalho',
    'essa série esta ficando sem graça, já devia ter acabado'
]

positive_classifications = []
negative_classifications = []

for i in range(len(positive_comments)):
    positive_classifications.append(1)

for i in range(len(negative_comments)):
    negative_classifications.append(0)

comments = positive_comments + negative_comments
classifications = positive_classifications + negative_classifications

data = {'tweet_text': comments, 'sentiment': classifications}
df = pd.DataFrame(data)

print("DATA FRAME: ", df)

df.to_csv('comentarios_gerados.csv', index=False, sep=';')