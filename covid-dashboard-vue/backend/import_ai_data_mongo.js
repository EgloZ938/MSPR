const { MongoClient } = require('mongodb');
const fs = require('fs');
const path = require('path');
const csv = require('csv-parser');
require('dotenv').config();

// Configuration MongoDB
const MONGO_URI = process.env.MONGO_URI;
const DB_NAME = process.env.DB_NAME;

class CovidAIDataImporter {
    constructor() {
        this.client = null;
        this.db = null;
        this.countriesMap = new Map();
        this.regionsMap = new Map();
    }

    async connect() {
        try {
            this.client = new MongoClient(MONGO_URI);
            await this.client.connect();
            this.db = this.client.db(DB_NAME);
            console.log('✅ Connecté à MongoDB pour l\'import des données IA');
        } catch (error) {
            console.error('❌ Erreur de connexion à MongoDB:', error);
            throw error;
        }
    }

    async disconnect() {
        if (this.client) {
            await this.client.close();
            console.log('🔌 Connexion MongoDB fermée');
        }
    }

    // Fonction pour lire un fichier CSV avec détection automatique du séparateur
    readCSV(filePath) {
        return new Promise((resolve, reject) => {
            const results = [];
            if (!fs.existsSync(filePath)) {
                console.warn(`⚠️  Fichier non trouvé: ${filePath}`);
                resolve([]);
                return;
            }

            // Lire le fichier pour détecter le séparateur
            const fileContent = fs.readFileSync(filePath, 'utf8');
            const firstLine = fileContent.split('\n')[0];

            // Détecter le séparateur le plus probable
            let separator = ',';
            const separators = [',', ';', '\t', '|'];
            let maxColumns = 0;

            separators.forEach(sep => {
                const columns = firstLine.split(sep).length;
                if (columns > maxColumns) {
                    maxColumns = columns;
                    separator = sep;
                }
            });

            console.log(`📖 Lecture de ${path.basename(filePath)} avec séparateur '${separator}'...`);

            fs.createReadStream(filePath)
                .pipe(csv({ separator }))
                .on('data', (data) => {
                    // Nettoyer les clés (supprimer les espaces)
                    const cleanedData = {};
                    Object.keys(data).forEach(key => {
                        const cleanKey = key.trim();
                        cleanedData[cleanKey] = data[key] ? data[key].trim() : data[key];
                    });
                    results.push(cleanedData);
                })
                .on('end', () => {
                    console.log(`✅ ${path.basename(filePath)}: ${results.length} lignes lues`);
                    resolve(results);
                })
                .on('error', (error) => {
                    console.error(`❌ Erreur lecture ${filePath}:`, error);
                    reject(error);
                });
        });
    }

    // Extraire la date du nom de fichier
    extractDateFromFilename(filename) {
        // Patterns de dates possibles dans les noms de fichiers
        const patterns = [
            /(\d{4}-\d{2}-\d{2})/,           // YYYY-MM-DD
            /(\d{4}_\d{2}_\d{2})/,           // YYYY_MM_DD
            /(\d{2}-\d{2}-\d{4})/,           // MM-DD-YYYY
            /(\d{2}_\d{2}_\d{4})/,           // MM_DD_YYYY
        ];

        for (const pattern of patterns) {
            const match = filename.match(pattern);
            if (match) {
                const dateStr = match[1];

                // Essayer de parser la date
                let parsedDate;
                if (dateStr.includes('-') || dateStr.includes('_')) {
                    const parts = dateStr.split(/[-_]/);

                    if (parts[0].length === 4) {
                        // Format YYYY-MM-DD
                        parsedDate = new Date(parts[0], parts[1] - 1, parts[2]);
                    } else {
                        // Format MM-DD-YYYY
                        parsedDate = new Date(parts[2], parts[0] - 1, parts[1]);
                    }
                }

                if (parsedDate && !isNaN(parsedDate.getTime())) {
                    return parsedDate;
                }
            }
        }

        console.warn(`⚠️  Impossible d'extraire la date du fichier: ${filename}`);
        return null;
    }

    // Nettoyer et normaliser les données
    cleanValue(value) {
        if (!value || value === '' || value === 'Unknown' || value === 'nan' || value === 'NA' || value === 'NULL') {
            return null;
        }

        // Essayer de convertir en nombre
        const numValue = parseFloat(value);
        if (!isNaN(numValue)) {
            return numValue;
        }

        return value.toString().trim();
    }

    parseDate(dateStr) {
        if (!dateStr || dateStr === '' || dateStr === 'Unknown' || dateStr === 'nan') {
            return null;
        }

        const date = new Date(dateStr);
        return isNaN(date.getTime()) ? null : date;
    }

    // Charger les mappings existants
    async loadExistingMappings() {
        console.log('📋 Chargement des mappings existants...');

        try {
            const countries = await this.db.collection('countries').find({}).toArray();
            countries.forEach(country => {
                this.countriesMap.set(country.country_name, country._id);
            });
            console.log(`📍 ${countries.length} pays chargés`);

            const regions = await this.db.collection('who_regions').find({}).toArray();
            regions.forEach(region => {
                this.regionsMap.set(region.region_name, region._id);
            });
            console.log(`🌍 ${regions.length} régions WHO chargées`);
        } catch (error) {
            console.warn('⚠️  Mappings existants non trouvés, création de nouveaux...');
        }
    }

    // Créer des mappings manquants
    async createMissingMappings(allData) {
        console.log('🔧 Création des mappings manquants...');

        const uniqueRegions = new Set();
        const uniqueCountries = new Set();

        // Parcourir toutes les données pour extraire régions et pays
        allData.forEach(fileData => {
            fileData.data.forEach(row => {
                if (row.region && row.region !== 'Unknown') {
                    uniqueRegions.add(row.region);
                }
                if (row.country && row.country !== 'Unknown') {
                    uniqueCountries.add(row.country);
                }
            });
        });

        // Créer les régions manquantes
        let regionId = (await this.db.collection('who_regions').countDocuments()) + 1;
        for (const region of uniqueRegions) {
            if (!this.regionsMap.has(region)) {
                await this.db.collection('who_regions').insertOne({
                    _id: regionId,
                    region_name: region
                });
                this.regionsMap.set(region, regionId);
                console.log(`➕ Région ajoutée: ${region}`);
                regionId++;
            }
        }

        // Créer les pays manquants
        let countryId = (await this.db.collection('countries').countDocuments()) + 1;
        for (const country of uniqueCountries) {
            if (!this.countriesMap.has(country)) {
                // Trouver la région pour ce pays
                const countryData = allData.find(fileData =>
                    fileData.data.some(row => row.country === country && row.region)
                );
                const regionName = countryData ?
                    countryData.data.find(row => row.country === country && row.region)?.region : null;
                const regionId = this.regionsMap.get(regionName) || null;

                await this.db.collection('countries').insertOne({
                    _id: countryId,
                    country_name: country,
                    continent: null,
                    population: null,
                    who_region_id: regionId
                });
                this.countriesMap.set(country, countryId);
                console.log(`➕ Pays ajouté: ${country}`);
                countryId++;
            }
        }

        console.log(`✅ Mappings créés: ${uniqueRegions.size} régions, ${uniqueCountries.size} pays`);
    }

    // Insertion massive des données démographiques par âge
    async insertAgeDemographics(allFileData) {
        console.log('📊 Insertion des données démographiques par âge...');

        // Nettoyer la collection existante
        await this.db.collection('age_demographics').deleteMany({});

        let demographicId = 1;
        let totalInserted = 0;
        const batchSize = 1000;
        let batch = [];

        // Traiter chaque fichier
        for (const fileInfo of allFileData) {
            const { filename, data, extractedDate } = fileInfo;
            console.log(`   📁 Traitement de ${filename} (${data.length} enregistrements)...`);

            for (const row of data) {
                // Vérifications de base
                if (!row.country || !row.age_group) {
                    continue;
                }

                // Filtrer les lignes "Total" 
                if (row.age_group && (
                    row.age_group.toLowerCase().includes('total') ||
                    row.age_group.toLowerCase().includes('all') ||
                    row.age_group === 'UNK'
                )) {
                    continue;
                }

                // Définir les tranches d'âge valides
                const validAgeGroups = [
                    '0-4', '5-14', '15-24', '25-34', '35-44',
                    '45-54', '55-64', '65-74', '75-84', '85+'
                ];

                // Si l'age_group n'est pas dans la liste, essayer de le normaliser
                let normalizedAgeGroup = row.age_group;
                if (!validAgeGroups.includes(row.age_group)) {
                    // Essayer quelques conversions communes
                    const ageGroupMappings = {
                        '0-9': '0-4',
                        '10-19': '15-24',
                        '20-29': '25-34',
                        '30-39': '35-44',
                        '40-49': '45-54',
                        '50-59': '55-64',
                        '60-69': '65-74',
                        '70-79': '75-84',
                        '80+': '85+',
                        '90+': '85+'
                    };

                    normalizedAgeGroup = ageGroupMappings[row.age_group];
                    if (!normalizedAgeGroup) {
                        // Si on ne peut pas normaliser, passer ce record
                        continue;
                    }
                }

                const countryId = this.countriesMap.get(row.country);
                const regionId = this.regionsMap.get(row.region);

                // Utiliser la date extraite du fichier ou celle du champ death_reference_date
                let referenceDate = extractedDate;
                if (!referenceDate && row.death_reference_date) {
                    referenceDate = this.parseDate(row.death_reference_date);
                }

                if (!referenceDate) {
                    console.warn(`⚠️  Pas de date de référence pour ${row.country} dans ${filename}`);
                    continue;
                }

                if (!countryId) {
                    console.warn(`⚠️  Pays non trouvé: ${row.country}`);
                    continue;
                }

                // Calculer les features IA
                const ageMapping = {
                    '0-4': 2, '5-14': 9, '15-24': 19, '25-34': 29, '35-44': 39,
                    '45-54': 49, '55-64': 59, '65-74': 69, '75-84': 79, '85+': 90
                };

                const ageNumeric = ageMapping[normalizedAgeGroup] || 50;
                let riskCategory = 'Medium';
                if (ageNumeric <= 18) riskCategory = 'Low';
                else if (ageNumeric <= 45) riskCategory = 'Medium';
                else if (ageNumeric <= 65) riskCategory = 'High';
                else riskCategory = 'Very_High';

                const cumDeathMale = this.cleanValue(row.cum_death_male) || 0;
                const cumDeathFemale = this.cleanValue(row.cum_death_female) || 0;
                const cumDeathBoth = this.cleanValue(row.cum_death_both) || (cumDeathMale + cumDeathFemale);

                // Calculer les taux de mortalité par sexe
                const totalDeaths = cumDeathMale + cumDeathFemale;
                const mortalityRateMale = totalDeaths > 0 ? cumDeathMale / totalDeaths : 0;
                const mortalityRateFemale = totalDeaths > 0 ? cumDeathFemale / totalDeaths : 0;

                const demographic = {
                    _id: demographicId,
                    country_id: countryId,
                    region_id: regionId,
                    country_name: row.country,
                    region_name: row.region,
                    country_code: this.cleanValue(row.country_code) || null,
                    age_group: normalizedAgeGroup,
                    age_numeric: ageNumeric,
                    risk_category: riskCategory,
                    death_reference_date: referenceDate,
                    year: referenceDate.getFullYear(),
                    month: referenceDate.getMonth() + 1,
                    week_of_year: this.getWeekNumber(referenceDate),
                    cum_death_male: cumDeathMale,
                    cum_death_female: cumDeathFemale,
                    cum_death_both: cumDeathBoth,
                    mortality_rate_male: mortalityRateMale,
                    mortality_rate_female: mortalityRateFemale,
                    pop_male: this.cleanValue(row.pop_male) || null,
                    pop_female: this.cleanValue(row.pop_female) || null,
                    pop_both: this.cleanValue(row.pop_both) || null,
                    source_file: filename,
                    created_at: new Date()
                };

                batch.push(demographic);
                demographicId++;

                // Insérer par batch pour optimiser les performances
                if (batch.length >= batchSize) {
                    await this.db.collection('age_demographics').insertMany(batch);
                    totalInserted += batch.length;
                    console.log(`      📈 ${totalInserted} enregistrements insérés...`);
                    batch = [];
                }
            }
        }

        // Insérer le dernier batch s'il reste des données
        if (batch.length > 0) {
            await this.db.collection('age_demographics').insertMany(batch);
            totalInserted += batch.length;
        }

        // Créer des index pour optimiser les performances
        console.log('🔍 Création des index...');
        await this.db.collection('age_demographics').createIndex({ country_id: 1, age_group: 1, death_reference_date: 1 });
        await this.db.collection('age_demographics').createIndex({ region_id: 1, age_group: 1 });
        await this.db.collection('age_demographics').createIndex({ age_numeric: 1, risk_category: 1 });
        await this.db.collection('age_demographics').createIndex({ death_reference_date: 1 });
        await this.db.collection('age_demographics').createIndex({ year: 1, month: 1 });

        console.log(`✅ ${totalInserted} enregistrements démographiques importés au total`);
    }

    // Utilitaire pour calculer la semaine de l'année
    getWeekNumber(date) {
        const startOfYear = new Date(date.getFullYear(), 0, 1);
        const days = Math.floor((date - startOfYear) / (24 * 60 * 60 * 1000));
        return Math.ceil((days + startOfYear.getDay() + 1) / 7);
    }

    // Créer des agrégations avancées pour l'IA
    async createAIAggregations() {
        console.log('🤖 Création des agrégations pour l\'IA...');

        // 1. Statistiques temporelles par pays et âge
        console.log('   📊 Agrégation 1: Séries temporelles par pays...');
        const timeSeriesCollection = this.db.collection('ai_time_series');
        await timeSeriesCollection.deleteMany({});

        const timeSeriesData = await this.db.collection('age_demographics').aggregate([
            {
                $group: {
                    _id: {
                        country_id: "$country_id",
                        country_name: "$country_name",
                        year: "$year",
                        month: "$month",
                        death_reference_date: "$death_reference_date"
                    },
                    total_deaths: { $sum: "$cum_death_both" },
                    total_male_deaths: { $sum: "$cum_death_male" },
                    total_female_deaths: { $sum: "$cum_death_female" },
                    avg_age: { $avg: "$age_numeric" },
                    age_groups_count: { $sum: 1 },
                    high_risk_deaths: {
                        $sum: {
                            $cond: [
                                { $in: ["$risk_category", ["High", "Very_High"]] },
                                "$cum_death_both",
                                0
                            ]
                        }
                    }
                }
            },
            {
                $addFields: {
                    mortality_trend: {
                        $cond: {
                            if: { $gt: ["$total_female_deaths", 0] },
                            then: { $divide: ["$total_male_deaths", "$total_female_deaths"] },
                            else: 0
                        }
                    },
                    high_risk_ratio: {
                        $cond: {
                            if: { $gt: ["$total_deaths", 0] },
                            then: { $divide: ["$high_risk_deaths", "$total_deaths"] },
                            else: 0
                        }
                    }
                }
            },
            { $sort: { "_id.death_reference_date": 1 } }
        ]).toArray();

        if (timeSeriesData.length > 0) {
            const indexedData = timeSeriesData.map((item, index) => ({
                _id: index + 1,
                ...item
            }));
            await timeSeriesCollection.insertMany(indexedData);
            console.log(`      ✅ ${timeSeriesData.length} séries temporelles créées`);
        }

        // 2. Matrice de corrélation âge-mortalité par région
        console.log('   🎯 Agrégation 2: Matrice de corrélation...');
        const correlationCollection = this.db.collection('ai_correlation_matrix');
        await correlationCollection.deleteMany({});

        const correlationData = await this.db.collection('age_demographics').aggregate([
            {
                $group: {
                    _id: {
                        region_name: "$region_name",
                        age_group: "$age_group",
                        risk_category: "$risk_category"
                    },
                    total_deaths: { $sum: "$cum_death_both" },
                    avg_mortality_male: { $avg: "$mortality_rate_male" },
                    avg_mortality_female: { $avg: "$mortality_rate_female" },
                    countries_affected: { $addToSet: "$country_name" },
                    data_points: { $sum: 1 }
                }
            },
            {
                $addFields: {
                    mortality_score: {
                        $multiply: [
                            { $add: ["$avg_mortality_male", "$avg_mortality_female"] },
                            { $divide: ["$total_deaths", 1000] }
                        ]
                    }
                }
            }
        ]).toArray();

        if (correlationData.length > 0) {
            const indexedCorrelation = correlationData.map((item, index) => ({
                _id: index + 1,
                ...item
            }));
            await correlationCollection.insertMany(indexedCorrelation);
            console.log(`      ✅ ${correlationData.length} éléments de corrélation créés`);
        }

        // 3. Features d'entraînement pour modèles ML
        console.log('   🧠 Agrégation 3: Features ML...');
        const mlFeaturesCollection = this.db.collection('ai_ml_features');
        await mlFeaturesCollection.deleteMany({});

        const mlFeatures = await this.db.collection('age_demographics').aggregate([
            {
                $addFields: {
                    // Calculer des features temporelles
                    day_of_year: { $dayOfYear: "$death_reference_date" },
                    quarter: {
                        $ceil: { $divide: ["$month", 3] }
                    },
                    season: {
                        $switch: {
                            branches: [
                                { case: { $in: ["$month", [12, 1, 2]] }, then: "Winter" },
                                { case: { $in: ["$month", [3, 4, 5]] }, then: "Spring" },
                                { case: { $in: ["$month", [6, 7, 8]] }, then: "Summer" },
                                { case: { $in: ["$month", [9, 10, 11]] }, then: "Fall" }
                            ],
                            default: "Unknown"
                        }
                    },
                    // Features démographiques
                    gender_death_ratio: {
                        $cond: {
                            if: { $gt: ["$cum_death_female", 0] },
                            then: { $divide: ["$cum_death_male", "$cum_death_female"] },
                            else: null
                        }
                    },
                    total_population: { $add: ["$pop_male", "$pop_female"] },
                    death_rate_per_pop: {
                        $cond: {
                            if: { $gt: ["$pop_both", 0] },
                            then: { $divide: ["$cum_death_both", "$pop_both"] },
                            else: 0
                        }
                    }
                }
            },
            {
                $project: {
                    country_id: 1,
                    country_name: 1,
                    region_name: 1,
                    age_group: 1,
                    age_numeric: 1,
                    risk_category: 1,
                    year: 1,
                    month: 1,
                    quarter: 1,
                    season: 1,
                    day_of_year: 1,
                    week_of_year: 1,
                    cum_death_both: 1,
                    cum_death_male: 1,
                    cum_death_female: 1,
                    gender_death_ratio: 1,
                    total_population: 1,
                    death_rate_per_pop: 1,
                    source_file: 1,
                    death_reference_date: 1
                }
            }
        ]).toArray();

        if (mlFeatures.length > 0) {
            const indexedFeatures = mlFeatures.map((item, index) => ({
                _id: index + 1,
                ...item
            }));

            // Insérer par batch
            const batchSize = 1000;
            for (let i = 0; i < indexedFeatures.length; i += batchSize) {
                const batch = indexedFeatures.slice(i, i + batchSize);
                await mlFeaturesCollection.insertMany(batch);
            }
            console.log(`      ✅ ${mlFeatures.length} features ML créées`);
        }

        // Créer des index pour les performances
        await timeSeriesCollection.createIndex({ "_id.country_id": 1, "_id.death_reference_date": 1 });
        await correlationCollection.createIndex({ "_id.region_name": 1, "_id.risk_category": 1 });
        await mlFeaturesCollection.createIndex({ country_id: 1, death_reference_date: 1 });
        await mlFeaturesCollection.createIndex({ age_group: 1, risk_category: 1 });

        console.log('✅ Toutes les agrégations IA créées avec succès!');
    }

    // Fonction principale d'import
    async importAIData() {
        try {
            console.log('🚀 Début de l\'importation COMPLÈTE des données IA COVID...');
            console.log('=' * 60);

            // Charger les mappings existants
            await this.loadExistingMappings();

            // Chemin vers les données nettoyées
            const dataPath = '../data/dataset_clean';

            if (!fs.existsSync(dataPath)) {
                throw new Error(`Le dossier ${dataPath} n'existe pas!`);
            }

            console.log('📂 Lecture de tous les fichiers de données...');
            const files = fs.readdirSync(dataPath);
            console.log(`📁 ${files.length} fichiers trouvés dans dataset_clean`);

            const allFileData = [];

            // Lire TOUS les fichiers CSV nettoyés
            for (const file of files) {
                if (file.endsWith('_clean.csv')) {
                    console.log(`🔍 Traitement de ${file}...`);

                    const filePath = path.join(dataPath, file);
                    const data = await this.readCSV(filePath);

                    if (data.length > 0) {
                        // Extraire la date du nom de fichier
                        const extractedDate = this.extractDateFromFilename(file);

                        // Ajouter les informations du fichier
                        allFileData.push({
                            filename: file,
                            data: data,
                            extractedDate: extractedDate,
                            recordCount: data.length
                        });

                        console.log(`   ✅ ${file}: ${data.length} enregistrements${extractedDate ? `, date: ${extractedDate.toISOString().split('T')[0]}` : ''}`);
                    } else {
                        console.log(`   ⚠️  ${file}: Aucune donnée`);
                    }
                }
            }

            if (allFileData.length === 0) {
                throw new Error('Aucun fichier de données trouvé!');
            }

            console.log('\n📊 Résumé des fichiers chargés:');
            console.log(`   • ${allFileData.length} fichiers traités`);
            console.log(`   • ${allFileData.reduce((sum, f) => sum + f.recordCount, 0)} enregistrements au total`);
            console.log(`   • ${allFileData.filter(f => f.extractedDate).length} fichiers avec dates extraites`);

            // Créer les mappings manquants
            await this.createMissingMappings(allFileData);

            // Importer toutes les données
            await this.insertAgeDemographics(allFileData);

            // Créer les agrégations pour l'IA
            await this.createAIAggregations();

            console.log('\n🎉 Import IA terminé avec succès!');
            console.log('📊 Collections créées:');
            console.log('   - age_demographics: Données démographiques détaillées avec features temporelles');
            console.log('   - ai_time_series: Séries temporelles par pays pour prédictions');
            console.log('   - ai_correlation_matrix: Matrices de corrélation âge-mortalité');
            console.log('   - ai_ml_features: Features préparées pour machine learning');
            console.log('\n🤖 Votre base de données est maintenant prête pour l\'entraînement IA!');

        } catch (error) {
            console.error('❌ Erreur lors de l\'import IA:', error);
            throw error;
        }
    }
}

// Fonction principale
async function main() {
    const importer = new CovidAIDataImporter();

    try {
        await importer.connect();
        await importer.importAIData();
    } catch (error) {
        console.error('💥 Erreur fatale:', error);
        process.exit(1);
    } finally {
        await importer.disconnect();
    }
}

// Exécuter si ce fichier est lancé directement
if (require.main === module) {
    main();
}

module.exports = CovidAIDataImporter;