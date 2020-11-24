from matplotlib import pyplot as plt





loss_G=[0.8891708019167878, 0.8614169276037882,0.8352864860513879, 0.8193176435839578, 0.8090083566608481, 0.8011409749127214, 0.7969727061424983, 0.7910099705168597, 0.7886147979819482, 0.7832691494060797, 0.7805533344154462, 0.7770289283357459, 0.7745453629246849, 0.7724424772756301, 0.7710419116942694, 0.7681789969878235, 0.7660616425142626, 0.7658133441810712, 0.7637349890103458, 0.7630339994092729, 0.7606043178963726, 0.7604705228467729, 0.7582094779781164, 0.7575396711884792, 0.7566592349018648, 0.7581246607310116, 0.7539201367453593, 0.7535044490803815, 0.7527340257525119, 0.7520619738004513, 0.7519970303985014, 0.7503376890593069, 0.7487493613760218, 0.7495308442076806, 0.7492808827912125, 0.7469789260739527, 0.7452794212736291, 0.7471301003437926, 0.7444597706807732, 0.7440963038306795, 0.7468926952060627, 0.7431629814958702, 0.7426252235183923, 0.742672870854266, 0.7429452838949676, 0.7455189052654547, 0.7402855148107544, 0.7438117502820589, 0.7396946213225902, 0.7293835881620402, 0.7274211498956914, 0.7266578778259111, 0.7265971752863164, 0.7260400423886666, 0.7232204510015753, 0.7248585880289935, 0.7244585333468154, 0.7229080408080722, 0.7232694288040702, 0.7216741993251873, 0.7216367799514646, 0.7217337377064884, 0.7212299897820164, 0.7204040194727095, 0.7201543906728968, 0.7204835980074932, 0.7189378452560882, 0.7199655475668426, 0.7187875025278866, 0.7194648763464322, 0.7177316109555943, 0.7188055469814373, 0.7170856694109758, 0.7170242184839066, 0.7179573745103883, 0.7162970353212279, 0.7162085592909145, 0.7168698843558413, 0.7155669417628151, 0.7161423685765071, 0.7160056631311734, 0.7152300842451464, 0.7152551968026226, 0.7141515747402929, 0.7154638305996679, 0.7147077929421407, 0.7153763524193205, 0.7146628896936734, 0.7144369598305518, 0.7119707736397309, 0.7149977515114101, 0.712803479436308, 0.7127923367783549, 0.7135921966802197, 0.7115503461874149, 0.7129917404632152, 0.7153912370146458, 0.7119799205977521, 0.7118961012006131, 0.7061487348592899, 0.7044156357767797, 0.7037257888336598, 0.703700426813692, 0.7032002545182646, 0.703202499680688, 0.7024349035943886, 0.7028491776385815, 0.7022585336129087, 0.7017665104255364, 0.7017865505790191, 0.701499336097156, 0.7012185244859077, 0.7004592438159911, 0.700749036076933, 0.7010260225966451, 0.7007641701347497, 0.701046395366783, 0.699619386761325, 0.7007792210384026, 0.6993527945120913, 0.6996161437489357, 0.6992989937680943, 0.6993297608087108, 0.6986667726605075, 0.6984143997732885, 0.699798085059392, 0.6992869364143393, 0.6988396501671066, 0.6986675210479819, 0.6983956900864271, 0.6976784022692438, 0.6975322588263155, 0.6981552082446355, 0.6991544718313607, 0.6973003418633983, 0.6978437959010985, 0.6979440798226754, 0.6978833772830807, 0.6971151575405526, 0.6983986004821611, 0.6970054356213812, 0.6971631374930816, 0.6967548505486845, 0.6969873495907485, 0.6967915631120146, 0.6966758540930476, 0.6963353377921705, 0.6958105102871679, 0.697469560586789, 0.6935200704215982, 0.691826261681497]
dice_validation=[0.30768852951102577, 0.4213287022069352,0.45469130132015934, 0.4677831349990162, 0.48189063780289987, 0.48539732064015906, 0.49980726525839686, 0.4526638676014531, 0.5239894044912065, 0.4792940375079657, 0.5183437189758958, 0.4822732493743038, 0.4148052937637387, 0.5023661693141616, 0.5086007178162927, 0.5187709307393101, 0.5124853847443802, 0.4821875694479889, 0.49139202655435715, 0.5244028555791113, 0.4297209997935226, 0.5129915426789183, 0.4794579827014913, 0.5091346198131111, 0.5072018535582652, 0.5012896025923748, 0.5308412247595264, 0.4519541406814367, 0.5159700053551728, 0.5187680957755597, 0.5121605093342799, 0.5099442162498704, 0.5275681304818924, 0.4810394641192724, 0.513348433461136, 0.5119664171485977, 0.4965990443140637, 0.5156229133660117, 0.49107242210614865, 0.5069642622966664, 0.5134461244850018, 0.5295661424776693, 0.5108529101706311, 0.5001503907101535, 0.4724734965942226, 0.5252660177010053, 0.5113057473233124, 0.49891007002921584, 0.5240731247630747, 0.5146846089593634, 0.5069063677462252, 0.48614807533796744, 0.5140047607168938, 0.5330230596623287, 0.506701349870051, 0.5231996389951284, 0.5158875723086744, 0.5178622044776586, 0.5126129857607391, 0.5223079656209126, 0.5139368700677021, 0.5217525369561251, 0.5111662930109805, 0.520413515842976, 0.5107543324745064, 0.5327380342740428, 0.5065503488343518, 0.5330171338136314, 0.4981452016555106, 0.49009000635098887, 0.5262754914810709, 0.4909586970411688, 0.5004151278591008, 0.49008229459954034, 0.5163243941778404, 0.5195430034681177, 0.5179555135682494, 0.5145045830190296, 0.5200016774417926, 0.5145216628157673, 0.5156076840673539, 0.5056305370593955, 0.5226336884322079, 0.4948488962858717, 0.5328361405991018, 0.5108761244748571, 0.5080509899409185, 0.5174642533389985, 0.5147836816051423, 0.4947803023950924, 0.5175489737917195, 0.5167112074558446, 0.5100437822998931, 0.5063301919756562, 0.5303811103541521, 0.5217104251332559, 0.5135649189948468, 0.5269312212167468, 0.4975077388409373, 0.514937640264361, 0.515625544243291, 0.5058339699911402, 0.49351926405624363, 0.4953756514018655, 0.4973345172385925, 0.49133270207994556, 0.5034535787007165, 0.5139981566207903, 0.5096251222373231, 0.5079753490131204, 0.5206689568897523, 0.5131954267819533, 0.5097415261980781, 0.5099342294625733, 0.5014183590849676, 0.5099539024051771, 0.5013522171015744, 0.5060993754584077, 0.5193213941148357, 0.5171752109588412, 0.5096211682501772, 0.5127335273954028, 0.5020925442178199, 0.5001416677213872, 0.507874167870755, 0.5039538832985708, 0.49363686525919515, 0.507837443551373, 0.503971256778268, 0.4995148133142691, 0.5125957700139449, 0.49863273900097055, 0.5137628947236672, 0.511086809061659, 0.5043935387171131, 0.5074176842058352, 0.5134253928607966, 0.4989144785088672, 0.5086886130741218, 0.497601011383403, 0.49634298078581107, 0.49763851195616343, 0.5019530352456699, 0.5090367809065629, 0.49903942854608896, 0.5046911711045156, 0.5080680761838086, 0.49902225112596565, 0.5025215437493998, 0.5118667570595851, 0.5135460211359297]

linewidth=2.0

plt.plot(loss_G,label='Generator loss',linewidth=linewidth)
plt.plot(dice_validation,label='Dice of validation set',linewidth=linewidth)
plt.legend(loc="lower right",labelspacing=0.5,handlelength=1, borderpad=0.4,fontsize=15)

#plt.show()

plt.savefig('generator_dice_1.png')