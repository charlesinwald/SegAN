from matplotlib import pyplot as plt





loss_G=[0.8891708019167878, 0.8614169276037882,0.8352864860513879, 0.8193176435839578, 0.8090083566608481, 0.8011409749127214, 0.7969727061424983, 0.7910099705168597, 0.7886147979819482, 0.7832691494060797, 0.7805533344154462, 0.7770289283357459, 0.7745453629246849, 0.7724424772756301, 0.7710419116942694, 0.7681789969878235, 0.7660616425142626, 0.7658133441810712, 0.7637349890103458, 0.7630339994092729, 0.7606043178963726, 0.7604705228467729, 0.7582094779781164, 0.7575396711884792, 0.7566592349018648, 0.7581246607310116, 0.7539201367453593, 0.7535044490803815, 0.7527340257525119, 0.7520619738004513, 0.7519970303985014, 0.7503376890593069, 0.7487493613760218, 0.7495308442076806, 0.7492808827912125, 0.7469789260739527, 0.7452794212736291, 0.7471301003437926, 0.7444597706807732, 0.7440963038306795, 0.7468926952060627, 0.7431629814958702, 0.7426252235183923, 0.742672870854266, 0.7429452838949676, 0.7455189052654547, 0.7402855148107544, 0.7438117502820589, 0.7396946213225902, 0.7293835881620402, 0.7274211498956914, 0.7266578778259111, 0.7265971752863164, 0.7260400423886666, 0.7232204510015753, 0.7248585880289935, 0.7244585333468154, 0.7229080408080722, 0.7232694288040702, 0.7216741993251873, 0.7216367799514646, 0.7217337377064884, 0.7212299897820164, 0.7204040194727095, 0.7201543906728968, 0.7204835980074932, 0.7189378452560882, 0.7199655475668426, 0.7187875025278866, 0.7194648763464322, 0.7177316109555943, 0.7188055469814373, 0.7170856694109758, 0.7170242184839066, 0.7179573745103883, 0.7162970353212279, 0.7162085592909145, 0.7168698843558413, 0.7155669417628151, 0.7161423685765071, 0.7160056631311734, 0.7152300842451464, 0.7152551968026226, 0.7141515747402929, 0.7154638305996679, 0.7147077929421407, 0.7153763524193205, 0.7146628896936734, 0.7144369598305518, 0.7119707736397309, 0.7149977515114101, 0.712803479436308, 0.7127923367783549, 0.7135921966802197, 0.7115503461874149, 0.7129917404632152, 0.7153912370146458, 0.7119799205977521, 0.7118961012006131, 0.7061487348592899, 0.7044156357767797, 0.7037257888336598, 0.703700426813692, 0.7032002545182646, 0.703202499680688, 0.7024349035943886, 0.7028491776385815, 0.7022585336129087, 0.7017665104255364, 0.7017865505790191, 0.701499336097156, 0.7012185244859077, 0.7004592438159911, 0.700749036076933, 0.7010260225966451, 0.7007641701347497, 0.701046395366783, 0.699619386761325, 0.7007792210384026, 0.6993527945120913, 0.6996161437489357, 0.6992989937680943, 0.6993297608087108, 0.6986667726605075, 0.6984143997732885, 0.699798085059392, 0.6992869364143393, 0.6988396501671066, 0.6986675210479819, 0.6983956900864271, 0.6976784022692438, 0.6975322588263155, 0.6981552082446355, 0.6991544718313607, 0.6973003418633983, 0.6978437959010985, 0.6979440798226754, 0.6978833772830807, 0.6971151575405526, 0.6983986004821611, 0.6970054356213812, 0.6971631374930816, 0.6967548505486845, 0.6969873495907485, 0.6967915631120146, 0.6966758540930476, 0.6963353377921705, 0.6958105102871679, 0.697469560586789, 0.6935200704215982, 0.691826261681497]
dice_validation=[0.30768852951102577, 0.4213287022069352,0.45469130132015934, 0.4677831349990162, 0.48189063780289987, 0.48539732064015906, 0.49980726525839686, 0.4526638676014531, 0.5239894044912065, 0.4792940375079657, 0.5183437189758958, 0.4822732493743038, 0.4148052937637387, 0.5023661693141616, 0.5086007178162927, 0.5187709307393101, 0.5124853847443802, 0.4821875694479889, 0.49139202655435715, 0.5244028555791113, 0.4297209997935226, 0.5129915426789183, 0.4794579827014913, 0.5091346198131111, 0.5072018535582652, 0.5012896025923748, 0.5308412247595264, 0.4519541406814367, 0.5159700053551728, 0.5187680957755597, 0.5121605093342799, 0.5099442162498704, 0.5275681304818924, 0.4810394641192724, 0.513348433461136, 0.5119664171485977, 0.4965990443140637, 0.5156229133660117, 0.49107242210614865, 0.5069642622966664, 0.5134461244850018, 0.5295661424776693, 0.5108529101706311, 0.5001503907101535, 0.4724734965942226, 0.5252660177010053, 0.5113057473233124, 0.49891007002921584, 0.5240731247630747, 0.5146846089593634, 0.5069063677462252, 0.48614807533796744, 0.5140047607168938, 0.5330230596623287, 0.506701349870051, 0.5231996389951284, 0.5158875723086744, 0.5178622044776586, 0.5126129857607391, 0.5223079656209126, 0.5139368700677021, 0.5217525369561251, 0.5111662930109805, 0.520413515842976, 0.5107543324745064, 0.5327380342740428, 0.5065503488343518, 0.5330171338136314, 0.4981452016555106, 0.49009000635098887, 0.5262754914810709, 0.4909586970411688, 0.5004151278591008, 0.49008229459954034, 0.5163243941778404, 0.5195430034681177, 0.5179555135682494, 0.5145045830190296, 0.5200016774417926, 0.5145216628157673, 0.5156076840673539, 0.5056305370593955, 0.5226336884322079, 0.4948488962858717, 0.5328361405991018, 0.5108761244748571, 0.5080509899409185, 0.5174642533389985, 0.5147836816051423, 0.4947803023950924, 0.5175489737917195, 0.5167112074558446, 0.5100437822998931, 0.5063301919756562, 0.5303811103541521, 0.5217104251332559, 0.5135649189948468, 0.5269312212167468, 0.4975077388409373, 0.514937640264361, 0.515625544243291, 0.5058339699911402, 0.49351926405624363, 0.4953756514018655, 0.4973345172385925, 0.49133270207994556, 0.5034535787007165, 0.5139981566207903, 0.5096251222373231, 0.5079753490131204, 0.5206689568897523, 0.5131954267819533, 0.5097415261980781, 0.5099342294625733, 0.5014183590849676, 0.5099539024051771, 0.5013522171015744, 0.5060993754584077, 0.5193213941148357, 0.5171752109588412, 0.5096211682501772, 0.5127335273954028, 0.5020925442178199, 0.5001416677213872, 0.507874167870755, 0.5039538832985708, 0.49363686525919515, 0.507837443551373, 0.503971256778268, 0.4995148133142691, 0.5125957700139449, 0.49863273900097055, 0.5137628947236672, 0.511086809061659, 0.5043935387171131, 0.5074176842058352, 0.5134253928607966, 0.4989144785088672, 0.5086886130741218, 0.497601011383403, 0.49634298078581107, 0.49763851195616343, 0.5019530352456699, 0.5090367809065629, 0.49903942854608896, 0.5046911711045156, 0.5080680761838086, 0.49902225112596565, 0.5025215437493998, 0.5118667570595851, 0.5135460211359297]

loss_G=[0.8871894580033776,0.8557654979617096, 0.8399747804153798, 0.8303823249284611, 0.8210441234499909, 0.8142458006393077, 0.8071159540220748, 0.8013578015704488, 0.7981060826501181, 0.7942520851312682, 0.7917884116949037, 0.7891974338265353, 0.7886097930198492, 0.7851492416026981, 0.7840072276980378, 0.7826570910076762, 0.7816314697265625, 0.7786177701728288, 0.7790798808253089, 0.767152653184048, 0.7654588388842206, 0.7661431778308957, 0.7635978876158248, 0.7627287576364916, 0.7611562152241551, 0.7618229888206305, 0.7603740248569223, 0.7587169381075127, 0.7577611346577489, 0.7495703586312228, 0.7480976193450218, 0.745958550031795, 0.7450124163960301, 0.7429700452227925, 0.7420196533203125, 0.741249261900436, 0.7401005500970885, 0.7400601852771848, 0.738564690878225, 0.7335118138512899]
dice_validation=[0.2703652581889025,0.3920269887085798, 0.386829195573809, 0.45705181274821827, 0.3845511302695327, 0.32193477631663525, 0.46856713617179757, 0.4778429797545277, 0.5007530976013804, 0.4937506256364534, 0.4643801845872436, 0.4586319270331969, 0.5027356578859471, 0.5030794641985288, 0.48383614107156503, 0.4911081474283258, 0.503476683475958, 0.4520257862827977, 0.5143436860447337, 0.5147264836543017, 0.5092059921829489, 0.5051498019404986, 0.5035145254253515, 0.49812943587503544, 0.5249068330312998, 0.5097902612636604, 0.518815991583703, 0.506407622416299, 0.5299063039155253, 0.501493722831152, 0.5092943803106854, 0.5179457197503694, 0.5160095874168295, 0.5327681250184363, 0.5247853584862274, 0.5184245611690286, 0.5280673430628231, 0.5207692128229535, 0.52754395953703, 0.5077652198485817]


dice_validation=[0.10737846341853981, 0.36721527278989324, 0.410857305122091, 0.4702218636847345,0.3610326889494733 , 0.4856720493951782, 0.5121241258554777, 0.4894073836189286, 0.5109786356743479, 0.47728952912030675, 0.5019022768840095, 0.5173914834864076, 0.504154709328168, 0.5080384140555968, 0.5103591462060256, 0.49594244857787906, 0.5084598672201416, 0.5160429745311755, 0.4912264964549486, 0.5349921740478102, 0.48937048568398284, 0.4940689982816251, 0.524908769274138, 0.5186935475735892, 0.5349970713911754, 0.5003049216759555, 0.5224813243212952, 0.5396769374062251, 0.5233424153820488, 0.5169074981508056, 0.5125768624005093, 0.48535095070214934, 0.5206914336587188, 0.5275845320753091, 0.5211046022805186, 0.5129717669291793, 0.517530170768257, 0.5358847656131516, 0.5110909602108352, 0.5230004077146564, 0.49606593665653587, 0.516966104505674, 0.5223992264625323, 0.5332129518854805, 0.5222580977649092, 0.5264442907052591, 0.5225213506136295, 0.4857884343371476, 0.48597311343533384, 0.5259658126151641, 0.5154359520002704, 0.5095293829421363, 0.5049555595675315, 0.5194486960180309, 0.5114877446404845, 0.5181075397322396, 0.5069665315932728, 0.47449362535493983, 0.5116316812532953, 0.5189644765422514, 0.5069360542691513, 0.5160984916312994, 0.5078609193676106, 0.5209808573095779, 0.5273366039636572, 0.5187430288429974, 0.4948809616210969, 0.505460086392042, 0.5265039677805241, 0.5051791985810767, 0.5048180733261086,0.5279063172841119, 0.5211313494669892, 0.5116113708590465, 0.5066049800746871, 0.503175658492312, 0.5125014956953707, 0.5093596040064754, 0.5090500810329797, 0.5201540428508856, 0.5039394597175552, 0.49735975777863267, 0.5123159498123401, 0.5206996904896327, 0.5073238152942596, 0.5210924459351927, 0.5006932978306522, 0.5286103826978524, 0.4988398434688203, 0.5012466893902449, 0.525842748963399, 0.5037529681845578, 0.5014165227507887, 0.5033692541088267, 0.4940737485668103, 0.5036842027414576, 0.49603863933090864, 0.5146526708644235, 0.5043775038443662, 0.49611909758730693, 0.5038077970057306, 0.5095603330495929, 0.50986926055761, 0.5107720299084697, 0.4975849168249304, 0.5081783389328376, 0.5190600621601306, 0.5022908685585797, 0.500091229579952, 0.5154480491713891, 0.5014604649635662, 0.4968918357629304, 0.500083101613821, 0.5156806848170529, 0.504937132669845, 0.49603052603188974, 0.5067307491397767, 0.5073654184332357, 0.514529510205133, 0.5016783301148752, 0.5047023889708814, 0.4962051815125599, 0.5007035292633636, 0.5170372174069158, 0.5095972017970798, 0.4927923063030706, 0.5045783807435822, 0.5149494428508831, 0.4960328020635269, 0.5138636788833606, 0.5048093828669511, 0.5038876877433056, 0.4985462415951892, 0.5058896726415583, 0.5041583154833903, 0.50074759829917, 0.4837605742813858, 0.5058918873913282, 0.5050207691481576, 0.49550180219301687, 0.48789360690028977, 0.4922491852787778, 0.5116259346230507, 0.4924103752218123, 0.5115723117428799, 0.4973864818929378, 0.5051546429277715, 0.4903478064777225, 0.4898816436089639, 0.49946716534138635, 0.49598169519471186, 0.49512117455840166, 0.49661533665319607, 0.5044299437709642, 0.49764305208548265, 0.4993032290999885, 0.49477145605349293, 0.5007165069669683, 0.4937834926844565, 0.4914444455545534, 0.5112593300839076, 0.49973061232766464, 0.4857822364563039, 0.49154189757144984, 0.4841324004896701, 0.4975460753018492, 0.4886667742079335, 0.4954994535880016, 0.49752951796201156, 0.489938931275588, 0.5024061987369713, 0.49356017607821595, 0.5014601475838306, 0.485323888888595, 0.500735524212401, 0.49193407853553117, 0.49226736233035345, 0.5030977653284565, 0.4908512732143435, 0.49850885826495295, 0.4933003374471991, 0.5026567260682455, 0.48952469926550773, 0.49193978421992623, 0.48135515877340773, 0.48946565804696446, 0.49890162902826585, 0.4918158630141324, 0.4936847102630717, 0.491962860748668, 0.48809559623419274, 0.48070252592809193, 0.49963620984368556, 0.4859510503645515, 0.5011565872263358, 0.49431724547205597, 0.48972316493912044, 0.4931140239399235, 0.48381705916744544, 0.49366230637430575, 0.4941435388362941, 0.4939465396226757, 0.48794241316552733, 0.4999823708806743, 0.4959411223841695, 0.49490066430576635, 0.4912045244925029, 0.4887239247929131, 0.48878853983794307, 0.4881859097902714, 0.48687585555974017, 0.4906303962137901, 0.4853994510934948, 0.4826183601377353, 0.4874691070437602, 0.4828106210053988, 0.48876117735562385, 0.4907714567376703, 0.4954660406476197, 0.48756790426971797, 0.48784220771957987, 0.48353119744971523, 0.4951901925421124, 0.48150660642746174, 0.48566224942113195, 0.49121450717879067]
loss_G=[0.8952974718670512, 0.8585905917855197, 0.8366046284520349, 0.8225339845169423, 0.8126907348632812, 0.8229292936103288, 0.8040646397790243, 0.8010520047919695, 0.797158440878225, 0.7930826586346293, 0.7898656268452489, 0.7888882659202399, 0.784838831701944, 0.7820674984954125, 0.7954776675202125, 0.7813569002373274, 0.7781098831531613, 0.7754780968954397, 0.7740428835846657, 0.7746601548305777, 0.7717360563056413, 0.7688306764114735, 0.7702604781749637, 0.7682972397915152, 0.7668266296386719, 0.7657376666401707, 0.7658819154251454, 0.7647804437681686, 0.7638983615609103, 0.7606275691542514, 0.7600784301757812, 0.7615061028059139, 0.7595348801723746, 0.7569019406340843, 0.7594489164130632, 0.7562951376271803, 0.7554710743039154, 0.756107862605605, 0.7541880940282067, 0.754618622535883, 0.7534805120423783, 0.7523358367210211, 0.7520793648653252, 0.7513510238292606, 0.7520764373069586, 0.7491565083348474, 0.7496671454851017, 0.7530810333961664, 0.7566820189010265, 0.7427072303239689, 0.7394027709960938, 0.7374130515165107, 0.7356055060098338, 0.7354511438414107, 0.7346092933832213, 0.7347380172374637, 0.7329512751379679, 0.7322809529858966, 0.7323077889375909, 0.7316203893617143, 0.731358949528184, 0.7305591494538063, 0.7303152306135311, 0.7303762657697811, 0.7283872116443723, 0.7300544117772302, 0.7289554241091706, 0.7289354634839434, 0.7291365335153979, 0.7273517874784248, 0.7266682025998138, 0.7260708919791288, 0.726399443870367, 0.7263941653939181, 0.7271770211153252, 0.724980820057004, 0.7262446381324945, 0.725716657416765, 0.7243804488071176, 0.7238078893617143, 0.7251712887786156, 0.723629308301349, 0.7232194057730741, 0.7236524182696675, 0.7244963535042697, 0.7237774604974792, 0.7232792432918105, 0.722685037657272, 0.7220761942309003, 0.7227642591609511, 0.7219637493754543, 0.7219102992567905, 0.7211996122848155, 0.7221966233364371, 0.721494852110397, 0.721029237259266, 0.7212710713231286, 0.7224615230116733, 0.7200457550758539, 0.7151437803756359, 0.7141223286473474, 0.7129388853561046, 0.7121542553568996, 0.7124710083007812, 0.713044100029524, 0.7128116696379906, 0.712360426437023, 0.7120225595873456, 0.7108371202335801, 0.7106482927189317, 0.7108099737832713, 0.7119740774465162, 0.711233937463095, 0.71080234438874, 0.7102348859920058, 0.7096014688181322, 0.7086996921273165, 0.7101760243260583, 0.7093347505081532, 0.7091481851976972, 0.7089497322259948, 0.7092540208683458, 0.7088751682015353, 0.7085594354673873, 0.7085783315259356, 0.7087042165357013, 0.7085087354793105, 0.7074114777321039, 0.708379213200059, 0.7081069059150163, 0.7083922541418741, 0.707444523656091, 0.707527338072311, 0.7076400934263717, 0.7075959139092024, 0.7066104800202125, 0.7080299909724745, 0.7069629403047784, 0.7071527436722157, 0.7063714848008267, 0.705590891283612, 0.7071202743885129, 0.7063601737798646, 0.7066047579743141, 0.7054395453874455, 0.7068892190622729, 0.7051880858665289, 0.7058248298112736, 0.7062180541282477, 0.7029268575269122, 0.7014816195465797, 0.7011545314345249, 0.7008507307185683, 0.7013009537098019, 0.7009259157402571, 0.7008278425349745, 0.7004705473434093, 0.7001082841740098, 0.6997475291407386, 0.700906842253929, 0.6999455828999364, 0.6999864800031795, 0.6998716398727062, 0.6996039013529933, 0.699854429378066, 0.6996976719346157, 0.6995378095050191, 0.699716612350109, 0.6991877444954806, 0.6995229055715162, 0.6988354172817496, 0.6992261132528615, 0.6990178573963254, 0.6987542440724928, 0.6989678671193678, 0.6991709775702898, 0.698532503704692, 0.698638073233671, 0.6980454644491506, 0.6982949279075445, 0.6985977971276571, 0.6985444800798283, 0.6984829126402389, 0.6979036552961483, 0.6980198704919149, 0.6984283979548964, 0.6978038965269576, 0.6976906532465026, 0.6979671744413154, 0.6980860066968341, 0.6974057042321493, 0.6969457227130269, 0.6977932952171149, 0.6973012879837391, 0.6971027462981468, 0.6980881801871366, 0.6974582672119141, 0.6986148301945176, 0.6968315479367279, 0.6955377889233966, 0.6948806407839753, 0.6947011725847111, 0.6947829667911973, 0.6945670815401299, 0.6946208865143532, 0.6943537246349246, 0.694580965263899, 0.6941385934519213, 0.6944662582042606, 0.6941574007965797, 0.6940576420273892, 0.6940626987191134, 0.6940393669660702, 0.6942030884498773, 0.6940947687903116, 0.6941629010577535, 0.6945185106854106, 0.6940528958342796, 0.6939192927160929, 0.6940880265346793, 0.6940583073815634, 0.6940300520076308, 0.6940215354742005, 0.6941700868828352, 0.693841623705487, 0.6937801893367324]

linewidth=2.0

plt.plot(loss_G,label='Generator loss',linewidth=linewidth)
plt.plot(dice_validation,label='Dice of validation set',linewidth=linewidth)
plt.legend(loc="lower right",labelspacing=0.5,handlelength=1, borderpad=0.4,fontsize=15)

#plt.show()

plt.savefig('generator_dice_l1.png')