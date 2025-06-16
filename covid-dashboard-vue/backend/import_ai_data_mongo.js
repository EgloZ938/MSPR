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

    // Fonction utilitaire pour lire un fichier CSV avec détection du séparateur
    readCSV(filePath) {
        return new Promise((resolve, reject) => {
            const results = [];
            if (!fs.existsSync(filePath)) {
                console.warn(`⚠️  Fichier non trouvé: ${filePath}`);
                resolve([]);
                return;
            }

            // Détecter le séparateur en lisant la première ligne
            const firstLine = fs.readFileSync(filePath, 'utf8').split('\n')[0];
            const separator = firstLine.includes(';') && firstLine.split(';').length > firstLine.split(',').length ? ';' : ',';

            fs.createReadStream(filePath)
                .pipe(csv({ separator }))
                .on('data', (data) => results.push(data))
                .on('end', () => {
                    console.log(`📖 Fichier IA lu: ${path.basename(filePath)} (${results.length} lignes, sep: '${separator}')`);
                    resolve(results);
                })
                .on('error', reject);
        });
    }

    // Nettoyer et normaliser les données
    cleanValue(value) {
        if (value === null || value === undefined || value === '' || value === 'Unknown' || value === 'nan' || value === 'NA') {
            return null;
        }
        if (!isNaN(value) && value !== '') {
            return parseFloat(value);
        }
        return value;
    }

    parseDate(dateStr) {
        if (!dateStr || dateStr === '' || dateStr === 'Unknown' || dateStr === 'nan') {
            return null;
        }
        const date = new Date(dateStr);
        return isNaN(date.getTime()) ? null : date;
    }

    // Charger les mappings existants depuis les collections existantes
    async loadExistingMappings() {
        console.log('📋 Chargement des mappings existants...');

        try {
            // Charger les pays existants
            const countries = await this.db.collection('countries').find({}).toArray();
            countries.forEach(country => {
                this.countriesMap.set(country.country_name, country._id);
            });
            console.log(`📍 ${countries.length} pays chargés`);

            // Charger les régions WHO existantes
            const regions = await this.db.collection('who_regions').find({}).toArray();
            regions.forEach(region => {
                this.regionsMap.set(region.region_name, region._id);
            });
            console.log(`🌍 ${regions.length} régions WHO chargées`);
        } catch (error) {
            console.warn('⚠️  Mappings existants non trouvés, création de nouveaux...');
        }
    }

    // Créer de nouveaux mappings si nécessaire
    async createMissingMappings(demographicData) {
        console.log('🔧 Création des mappings manquants...');

        // Collecter toutes les régions et pays uniques
        const uniqueRegions = new Set();
        const uniqueCountries = new Set();

        demographicData.forEach(dataArray => {
            dataArray.forEach(row => {
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
                regionId++;
            }
        }

        // Créer les pays manquants
        let countryId = (await this.db.collection('countries').countDocuments()) + 1;
        for (const country of uniqueCountries) {
            if (!this.countriesMap.has(country)) {
                // Trouver la région pour ce pays
                const countryRegion = demographicData.flat().find(row => row.country === country)?.region;
                const regionId = this.regionsMap.get(countryRegion) || null;

                await this.db.collection('countries').insertOne({
                    _id: countryId,
                    country_name: country,
                    continent: null,
                    population: null,
                    who_region_id: regionId
                });
                this.countriesMap.set(country, countryId);
                countryId++;
            }
        }

        console.log(`✅ Mappings créés: ${uniqueRegions.size} régions, ${uniqueCountries.size} pays`);
    }

    // Insertion des données démographiques par âge
    async insertAgeDemographics(demographicFiles) {
        console.log('📊 Insertion des données démographiques par âge...');
        const collection = this.db.collection('age_demographics');
        await collection.deleteMany({});

        let demographicId = 1;
        let totalInserted = 0;

        for (const fileData of demographicFiles) {
            console.log(`   📁 Traitement de ${fileData.length} enregistrements démographiques...`);

            const validDemographics = [];

            for (const row of fileData) {
                // Vérifier que les données essentielles sont présentes
                if (!row.country || !row.age_group || !row.death_reference_date) {
                    continue;
                }

                // Filtrer les lignes "Total" qui ne sont pas utiles pour l'IA
                if (row.age_group && (row.age_group.includes('Total') || row.age_group.includes('total'))) {
                    continue;
                }

                // Filtrer les tranches d'âge valides
                const validAgeGroups = ['0-4', '5-14', '15-24', '25-34', '35-44', '45-54', '55-64', '65-74', '75-84', '85+'];
                if (!validAgeGroups.includes(row.age_group)) {
                    continue;
                }

                const countryId = this.countriesMap.get(row.country);
                const regionId = this.regionsMap.get(row.region);
                const date = this.parseDate(row.death_reference_date);

                if (countryId && date) {
                    // Calculer les features IA
                    const ageMapping = {
                        '0-4': 2, '5-14': 9, '15-24': 19, '25-34': 29, '35-44': 39,
                        '45-54': 49, '55-64': 59, '65-74': 69, '75-84': 79, '85+': 90
                    };

                    const ageNumeric = ageMapping[row.age_group] || 50;
                    let riskCategory = 'Medium';
                    if (ageNumeric <= 18) riskCategory = 'Low';
                    else if (ageNumeric <= 45) riskCategory = 'Medium';
                    else if (ageNumeric <= 65) riskCategory = 'High';
                    else riskCategory = 'Very_High';

                    const cumDeathMale = this.cleanValue(row.cum_death_male) || 0;
                    const cumDeathFemale = this.cleanValue(row.cum_death_female) || 0;
                    const cumDeathBoth = this.cleanValue(row.cum_death_both) || cumDeathMale + cumDeathFemale;

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
                        country_code: row.country_code || null,
                        age_group: row.age_group,
                        age_numeric: ageNumeric,
                        risk_category: riskCategory,
                        death_reference_date: date,
                        year: date.getFullYear(),
                        month: date.getMonth() + 1,
                        week_of_year: this.getWeekNumber(date),
                        cum_death_male: cumDeathMale,
                        cum_death_female: cumDeathFemale,
                        cum_death_both: cumDeathBoth,
                        mortality_rate_male: mortalityRateMale,
                        mortality_rate_female: mortalityRateFemale,
                        pop_male: this.cleanValue(row.pop_male) || null,
                        pop_female: this.cleanValue(row.pop_female) || null,
                        pop_both: this.cleanValue(row.pop_both) || null,
                        source_file: path.basename(row._filename || 'unknown'),
                        created_at: new Date()
                    };

                    validDemographics.push(demographic);
                    demographicId++;
                }
            }

            // Insertion par lots pour optimiser les performances
            if (validDemographics.length > 0) {
                const batchSize = 1000;
                for (let i = 0; i < validDemographics.length; i += batchSize) {
                    const batch = validDemographics.slice(i, i + batchSize);
                    await collection.insertMany(batch);
                    totalInserted += batch.length;
                    console.log(`      📈 Batch ${Math.floor(i / batchSize) + 1}/${Math.ceil(validDemographics.length / batchSize)} inséré`);
                }
            }
        }

        // Créer des index pour les performances
        await collection.createIndex({ country_id: 1, age_group: 1, death_reference_date: 1 });
        await collection.createIndex({ region_id: 1, age_group: 1 });
        await collection.createIndex({ age_numeric: 1, risk_category: 1 });
        await collection.createIndex({ death_reference_date: 1 });

        console.log(`✅ ${totalInserted} enregistrements démographiques importés`);
    }

    // Utilitaire pour calculer la semaine de l'année
    getWeekNumber(date) {
        const startOfYear = new Date(date.getFullYear(), 0, 1);
        const days = Math.floor((date - startOfYear) / (24 * 60 * 60 * 1000));
        return Math.ceil((days + startOfYear.getDay() + 1) / 7);
    }

    // Créer des agrégations pour l'IA
    async createAIAggregations() {
        console.log('🤖 Création des agrégations pour l\'IA...');

        // Agrégation 1: Statistiques par tranche d'âge et sexe
        const ageStatsCollection = this.db.collection('ai_age_statistics');
        await ageStatsCollection.deleteMany({});

        const ageStats = await this.db.collection('age_demographics').aggregate([
            {
                $group: {
                    _id: {
                        age_group: "$age_group",
                        age_numeric: "$age_numeric",
                        risk_category: "$risk_category"
                    },
                    total_male_deaths: { $sum: "$cum_death_male" },
                    total_female_deaths: { $sum: "$cum_death_female" },
                    total_deaths: { $sum: "$cum_death_both" },
                    avg_mortality_rate_male: { $avg: "$mortality_rate_male" },
                    avg_mortality_rate_female: { $avg: "$mortality_rate_female" },
                    record_count: { $sum: 1 },
                    countries_affected: { $addToSet: "$country_name" },
                    latest_date: { $max: "$death_reference_date" }
                }
            },
            {
                $addFields: {
                    mortality_ratio: {
                        $cond: {
                            if: { $gt: ["$total_female_deaths", 0] },
                            then: { $divide: ["$total_male_deaths", "$total_female_deaths"] },
                            else: 0
                        }
                    }
                }
            }
        ]).toArray();

        if (ageStats.length > 0) {
            await ageStatsCollection.insertMany(ageStats.map((stat, index) => ({
                _id: index + 1,
                ...stat
            })));
            console.log(`   📊 ${ageStats.length} statistiques par âge créées`);
        }

        // Agrégation 2: Matrice de risque par région et âge pour l'IA
        const riskMatrixCollection = this.db.collection('ai_risk_matrix');
        await riskMatrixCollection.deleteMany({});

        const riskMatrix = await this.db.collection('age_demographics').aggregate([
            {
                $group: {
                    _id: {
                        region_name: "$region_name",
                        risk_category: "$risk_category",
                        age_group: "$age_group"
                    },
                    total_deaths: { $sum: "$cum_death_both" },
                    male_deaths: { $sum: "$cum_death_male" },
                    female_deaths: { $sum: "$cum_death_female" },
                    countries_in_category: { $addToSet: "$country_name" },
                    avg_age_numeric: { $avg: "$age_numeric" }
                }
            },
            {
                $addFields: {
                    risk_score: {
                        $multiply: [
                            { $divide: ["$total_deaths", { $add: ["$total_deaths", 1] }] },
                            { $multiply: ["$avg_age_numeric", 0.01] }
                        ]
                    }
                }
            }
        ]).toArray();

        if (riskMatrix.length > 0) {
            await riskMatrixCollection.insertMany(riskMatrix.map((matrix, index) => ({
                _id: index + 1,
                ...matrix
            })));
            console.log(`   🎯 ${riskMatrix.length} éléments de matrice de risque créés`);
        }

        // Créer des index pour les performances
        await ageStatsCollection.createIndex({ "_id.age_group": 1, "_id.risk_category": 1 });
        await riskMatrixCollection.createIndex({ "_id.region_name": 1, "_id.risk_category": 1 });
    }

    // Fonction principale d'import des données IA
    async importAIData() {
        try {
            console.log('🚀 Début de l\'importation des données IA COVID...');

            // Charger les mappings existants
            await this.loadExistingMappings();

            // Chemins des fichiers nettoyés
            const dataPath = '../data/dataset_clean';

            // Lire tous les fichiers démographiques nettoyés
            console.log('📖 Lecture des fichiers démographiques...');
            const demographicFiles = [];

            const files = fs.readdirSync(dataPath);
            console.log(`📁 ${files.length} fichiers trouvés dans dataset_clean`);

            for (const file of files) {
                // Identifier les fichiers démographiques nettoyés
                if ((file.includes('cum_deaths_by_age_sex') ||
                    file.includes('covid_pooled') ||
                    file.includes('pooled_AS')) &&
                    file.includes('clean') &&
                    file.endsWith('.csv')) {

                    console.log(`   🔍 Lecture de ${file}...`);
                    const data = await this.readCSV(path.join(dataPath, file));
                    if (data.length > 0) {
                        // Ajouter le nom du fichier pour traçabilité
                        data.forEach(row => row._filename = file);
                        demographicFiles.push(data);
                    }
                }
            }

            if (demographicFiles.length === 0) {
                console.error('❌ Aucun fichier démographique trouvé !');
                console.log('💡 Vérifiez que les fichiers nettoyés existent dans dataset_clean/');
                return;
            }

            console.log(`✅ ${demographicFiles.length} fichiers démographiques trouvés`);

            // Créer les mappings manquants
            await this.createMissingMappings(demographicFiles);

            // Import des données dans l'ordre
            await this.insertAgeDemographics(demographicFiles);
            await this.createAIAggregations();

            console.log('🎉 Import des données IA terminé avec succès!');
            console.log('📊 Collections créées :');
            console.log('   - age_demographics : Données démographiques détaillées');
            console.log('   - ai_age_statistics : Statistiques par âge pour l\'IA');
            console.log('   - ai_risk_matrix : Matrice de risque pour l\'IA');

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