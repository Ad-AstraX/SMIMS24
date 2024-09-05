
def healing():
        output = 0
        if output == 0:
            return("Dein Apfelbaum hat: 'Apple scab'.   'Apple scab wird durch eine Pilzart verursacht. Folgendes kannst du tun, um deinem Apfelbaum zu helfen: Neben der Wahl der Sorte gibt es eine Reihe von vorbeugenden Maßnahmen, die den Befall zumindest reduzieren können. Dabei werden einerseits das Falllaub verringert durch Blattspritzungen mit Blattdüngern oder Zerkleinerung des Falllaubes durch Mulchen. Andererseits wird durch Schnitt und Erziehungsform der Baumkrone eine gute Belüftung und schnelleres Abtrocknen der Blätter gesichert, die Infektionsbedingungen von Nässe und Temperatur werden so beeinflusst. Durch Förderung eines zeitigen Triebabschlusses werden zudem Neuinfektionen an jungen empfindlichen Blättern bis weit in den Spätsommer verhindert. Im ökologischen Landbau sind als direkte Maßnahmen Schwefel und Kupferverbindungen wie zum Beispiel Kupferoxychlorid als Pflanzenschutzmittel zugelassen. Auch der Schutz der Bäume von äußerer Feuchtigkeit wie Regen erweist sich als wirkungsvoll[3]. In diesem Zusammenhang gilt auch die Nutzung von Agri-PV zum Schutz vor Niederschlägen als vielversprechend. Quelle: Wikipedia")
        elif output == 1:
            return('Dein Apfelbaum hat: "Black rot".Black rot wird durch eine Pilzart verursacht. Folgendes kannst du tun, um deinem Apfelbaum zu helfen:'
            'Befallene Reborgane sind als Ausgangsinokulum für weitere Infektionen aus den Rebanlagen zu entfernen. Weiterhin müssen nicht mehr bewirtschaftete Drieschen konsequent gerodet werden. Im integrierten Weinbau haben sich Pflanzenschutzmittel aus den Wirkstoffklassen der Strobilurine, Triazole und Dithiocarbamate als sehr wirkungsvoll herausgestellt. Ein besonderes Problem stellt die Schwarzfäule im ökologischen Weinbau dar. Die Grundlage für eine erfolgreiche Schwarzfäule-Bekämpfungsstrategie bilden hier kulturtechnische Maßnahmen. Von den direkten Bekämpfungsmaßnahmen hat sich der wöchentliche Einsatz der Kombination Schwefel-Kupfer als am Wirkungsvollsten herausgestellt.'
            'Quelle: Wikipedia')
        elif output == 2:
            suggestion = 'Dein Apfelbaum hat: "Cedar apple rust".\n\nCedar apple rust wird durch eine Pilzart verursacht. \nFolgendes kannst du tun, um deinem Apfelbaum zu helfen: Wenn Sie Anzeichen von Cedar apple rust an Ihren Pflanzen bemerken, entfernen und vernichten Sie alle infizierten Zweige.\nAußerdem gibt es Biofungizidprodukte, die bei der Bekämpfung helfen können. Darunter z.B. Serenade garden oder Regalia. \n\nUnbezahlte Werbung, Quelle: Gardenia'
            return suggestion
        else:
            return('error')
