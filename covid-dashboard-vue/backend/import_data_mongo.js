const { MongoClient } = require('mongodb');
const fs = require('fs');
const path = require('path');
const csv = require('csv-parser');
require('dotenv').config();

const { execSync } = require('child_process');

// Configuration MongoDB
const MONGO_URI = process.env.MONGO_URI;
const DB_NAME = process.env.DB_NAME;

class CovidDataImporter {
    constructor() {
        this.client = null;
        this.db = null;
        this.whoRegionsMap = new Map();
        this.countriesMap = new Map();
        this.provincesMap = new Map();
        this.countiesMap = new Map();
    }

    async connect() {
        try {
            this.client = new MongoClient(MONGO_URI);
            await this.client.connect();
            this.db = this.client.db(DB_NAME);
            console.log('✅ Connecté à MongoDB avec succès!');
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

    // Fonction utilitaire pour lire un fichier CSV
    readCSV(filePath) {
        return new Promise((resolve, reject) => {
            const results = [];
            if (!fs.existsSync(filePath)) {
                console.warn(`⚠️  Fichier non trouvé: ${filePath}`);
                resolve([]);
                return;
            }

            fs.createReadStream(filePath)
                .pipe(csv())
                .on('data', (data) => results.push(data))
                .on('end', () => {
                    console.log(`📖 Fichier lu: ${path.basename(filePath)} (${results.length} lignes)`);
                    resolve(results);
                })
                .on('error', reject);
        });
    }

    // Nettoyer et normaliser les données (exactement comme PostgreSQL)
    cleanValue(value) {
        if (value === null || value === undefined || value === '' || value === 'Unknown') {
            return null;
        }
        if (!isNaN(value) && value !== '') {
            return parseFloat(value);
        }
        return value;
    }

    parseDate(dateStr) {
        if (!dateStr || dateStr === '' || dateStr === 'Unknown') {
            return null;
        }
        const date = new Date(dateStr);
        return isNaN(date.getTime()) ? null : date;
    }

    // 1. Insertion des régions WHO (équivalent PostgreSQL)
    async insertWHORegions(df) {
        console.log('🌍 Insertion des régions WHO...');
        const collection = this.db.collection('who_regions');
        await collection.deleteMany({});

        const regions = [...new Set(df.map(row => row['WHO Region']).filter(Boolean))];

        for (let i = 0; i < regions.length; i++) {
            const region = regions[i];
            const doc = {
                _id: i + 1, // ID numérique comme PostgreSQL
                region_name: region
            };
            await collection.insertOne(doc);
            this.whoRegionsMap.set(region, i + 1);
        }

        console.log(`✅ ${regions.length} régions WHO importées`);
    }

    // 2. Insertion des pays (exactement comme PostgreSQL)
    async insertCountries(dfCountries, dfWorldometer) {
        console.log('🏳️ Insertion des pays...');
        const collection = this.db.collection('countries');
        await collection.deleteMany({});

        let countryId = 1;
        for (const row of dfCountries) {
            const worldometerData = dfWorldometer.find(w =>
                w['Country/Region'] === row['Country/Region']
            );

            const country = {
                _id: countryId,
                country_name: row['Country/Region'],
                continent: worldometerData ? this.cleanValue(worldometerData['Continent']) : null,
                population: worldometerData ? this.cleanValue(worldometerData['Population']) : null,
                who_region_id: this.whoRegionsMap.get(row['WHO Region']) || null
            };

            await collection.insertOne(country);
            this.countriesMap.set(row['Country/Region'], countryId);
            countryId++;
        }

        console.log(`✅ ${dfCountries.length} pays importés`);
    }

    // 3. Insertion des provinces (exactement comme PostgreSQL)
    async insertProvinces(dfProvinces) {
        console.log('🗾 Insertion des provinces...');
        const collection = this.db.collection('provinces');
        await collection.deleteMany({});

        let provinceId = 1;
        const seen = new Set();

        for (const row of dfProvinces) {
            if (row['Province/State'] && row['Country/Region']) {
                const key = `${row['Country/Region']}_${row['Province/State']}`;
                if (!seen.has(key)) {
                    seen.add(key);

                    const province = {
                        _id: provinceId,
                        province_name: row['Province/State'],
                        country_id: this.countriesMap.get(row['Country/Region']) || null,
                        latitude: this.cleanValue(row['Lat']),
                        longitude: this.cleanValue(row['Long'])
                    };

                    await collection.insertOne(province);
                    this.provincesMap.set(key, provinceId);
                    provinceId++;
                }
            }
        }

        console.log(`✅ ${provinceId - 1} provinces importées`);
    }

    // 4. Insertion des comtés US (exactement comme PostgreSQL)
    async insertUSCounties(dfCounties) {
        console.log('🏛️ Insertion des comtés US...');
        const collection = this.db.collection('us_counties');
        await collection.deleteMany({});

        let countyId = 1;
        const seen = new Set();

        for (const row of dfCounties) {
            if (row['FIPS'] && !seen.has(row['FIPS'])) {
                seen.add(row['FIPS']);

                // Trouver l'état correspondant
                const stateKey = `US_${row['Province_State']}`;
                const stateId = this.provincesMap.get(stateKey);

                if (stateId) {
                    const county = {
                        _id: countyId,
                        county_name: row['Admin2'],
                        state_id: stateId,
                        fips: String(parseInt(row['FIPS'])),
                        latitude: this.cleanValue(row['Lat']),
                        longitude: this.cleanValue(row['Long_'])
                    };

                    await collection.insertOne(county);
                    this.countiesMap.set(county.fips, countyId);
                    countyId++;
                }
            }
        }

        console.log(`✅ ${countyId - 1} comtés US importés`);
    }

    // 5. Insertion des statistiques quotidiennes (exactement comme PostgreSQL)
    async insertDailyStats(dfFull) {
        console.log('📊 Insertion des statistiques quotidiennes par pays...');
        const collection = this.db.collection('daily_stats');
        await collection.deleteMany({});

        let statsId = 1;
        const validStats = [];

        for (const row of dfFull) {
            const countryId = this.countriesMap.get(row['Country/Region']);
            const date = this.parseDate(row['Date']);

            if (countryId && date) {
                const stat = {
                    _id: statsId,
                    country_id: countryId,
                    date: date,
                    confirmed: this.cleanValue(row['Confirmed']) || 0,
                    deaths: this.cleanValue(row['Deaths']) || 0,
                    recovered: this.cleanValue(row['Recovered']) || 0,
                    active: this.cleanValue(row['Active']) || 0,
                    new_cases: this.cleanValue(row['New cases']) || 0,
                    new_deaths: this.cleanValue(row['New deaths']) || 0,
                    new_recovered: this.cleanValue(row['New recovered']) || 0
                };

                validStats.push(stat);
                statsId++;
            }
        }

        // Insertion par lots
        const batchSize = 1000;
        for (let i = 0; i < validStats.length; i += batchSize) {
            const batch = validStats.slice(i, i + batchSize);
            await collection.insertMany(batch);
            console.log(`   📈 Batch ${Math.floor(i / batchSize) + 1}/${Math.ceil(validStats.length / batchSize)} inséré`);
        }

        // Index pour les performances
        await collection.createIndex({ country_id: 1, date: 1 });
        console.log(`✅ ${validStats.length} statistiques quotidiennes importées`);
    }

    // 6. Insertion des statistiques par province (exactement comme PostgreSQL)
    async insertProvinceStats(dfComplete) {
        console.log('📊 Insertion des statistiques par province...');
        const collection = this.db.collection('province_stats');
        await collection.deleteMany({});

        let statsId = 1;
        const validStats = [];

        for (const row of dfComplete) {
            if (row['Province/State']) {
                const provinceKey = `${row['Country/Region']}_${row['Province/State']}`;
                const provinceId = this.provincesMap.get(provinceKey);
                const date = this.parseDate(row['Date']);

                if (provinceId && date) {
                    const stat = {
                        _id: statsId,
                        province_id: provinceId,
                        date: date,
                        confirmed: this.cleanValue(row['Confirmed']) || 0,
                        deaths: this.cleanValue(row['Deaths']) || 0,
                        recovered: this.cleanValue(row['Recovered']) || 0,
                        active: this.cleanValue(row['Active']) || 0
                    };

                    validStats.push(stat);
                    statsId++;
                }
            }
        }

        // Insertion par lots
        const batchSize = 1000;
        for (let i = 0; i < validStats.length; i += batchSize) {
            const batch = validStats.slice(i, i + batchSize);
            await collection.insertMany(batch);
        }

        await collection.createIndex({ province_id: 1, date: 1 });
        console.log(`✅ ${validStats.length} statistiques par province importées`);
    }

    // 7. Insertion des statistiques par comté US (exactement comme PostgreSQL)
    async insertCountyStats(dfCounties) {
        console.log('📊 Insertion des statistiques par comté US...');
        const collection = this.db.collection('county_stats');
        await collection.deleteMany({});

        let statsId = 1;
        const validStats = [];

        for (const row of dfCounties) {
            if (row['FIPS']) {
                const fips = String(parseInt(row['FIPS']));
                const countyId = this.countiesMap.get(fips);
                const date = this.parseDate(row['Date']);

                if (countyId && date) {
                    const stat = {
                        _id: statsId,
                        county_id: countyId,
                        date: date,
                        confirmed: this.cleanValue(row['Confirmed']) || 0,
                        deaths: this.cleanValue(row['Deaths']) || 0
                    };

                    validStats.push(stat);
                    statsId++;
                }
            }
        }

        // Insertion par lots
        const batchSize = 1000;
        for (let i = 0; i < validStats.length; i += batchSize) {
            const batch = validStats.slice(i, i + batchSize);
            await collection.insertMany(batch);
        }

        await collection.createIndex({ county_id: 1, date: 1 });
        console.log(`✅ ${validStats.length} statistiques par comté importées`);
    }

    // 8. Insertion des détails par pays (exactement comme PostgreSQL)
    async insertCountryDetails(dfWorldometer) {
        console.log('📋 Insertion des détails par pays...');
        const collection = this.db.collection('country_details');
        await collection.deleteMany({});

        let detailsId = 1;
        const currentDate = new Date();

        for (const row of dfWorldometer) {
            const countryId = this.countriesMap.get(row['Country/Region']);

            if (countryId) {
                const detail = {
                    _id: detailsId,
                    country_id: countryId,
                    total_tests: this.cleanValue(row['TotalTests']),
                    tests_per_million: this.cleanValue(row['Tests/1M pop']),
                    cases_per_million: this.cleanValue(row['Tot Cases/1M pop']),
                    deaths_per_million: this.cleanValue(row['Deaths/1M pop']),
                    serious_critical: this.cleanValue(row['Serious,Critical']),
                    last_updated: currentDate
                };

                await collection.insertOne(detail);
                detailsId++;
            }
        }

        await collection.createIndex({ country_id: 1, last_updated: 1 });
        console.log(`✅ ${detailsId - 1} détails de pays importés`);
    }

    // Fonction principale d'import (respecte l'ordre PostgreSQL)
    async importAll() {
        try {
            console.log('🚀 Début de l\'importation des données COVID...');
            console.log('📁 Nettoyage des fichiers CSV d\'abord...');

            // Exécuter le script Python de nettoyage
            try {
                execSync('python data_cleaner.py', { cwd: '../backend', stdio: 'inherit' });
            } catch (error) {
                console.warn('⚠️  Impossible d\'exécuter data_cleaner.py automatiquement');
                console.log('💡 Assurez-vous que les fichiers CSV nettoyés existent dans ../data/dataset_clean/');
            }

            // Chemins des fichiers (exactement comme PostgreSQL)
            const dataPath = '../data/dataset_clean';

            // Lecture des fichiers CSV
            console.log('📖 Lecture des fichiers CSV...');
            const dfCountryWise = await this.readCSV(path.join(dataPath, 'country_wise_latest_clean.csv'));
            const dfFullGrouped = await this.readCSV(path.join(dataPath, 'full_grouped_clean.csv'));
            const dfWorldometer = await this.readCSV(path.join(dataPath, 'worldometer_data_clean.csv'));
            const dfComplete = await this.readCSV(path.join(dataPath, 'covid_19_clean_complete_clean.csv'));
            const dfCounties = await this.readCSV(path.join(dataPath, 'usa_county_wise_clean.csv'));

            // Import des données dans l'ordre exact PostgreSQL
            await this.insertWHORegions(dfCountryWise);
            await this.insertCountries(dfCountryWise, dfWorldometer);
            await this.insertProvinces(dfComplete);
            await this.insertUSCounties(dfCounties);
            await this.insertDailyStats(dfFullGrouped);
            await this.insertProvinceStats(dfComplete);
            await this.insertCountyStats(dfCounties);
            await this.insertCountryDetails(dfWorldometer);

            console.log('🎉 Import terminé avec succès!');
            console.log('📊 Toutes les données sont maintenant identiques à PostgreSQL');

        } catch (error) {
            console.error('❌ Erreur lors de l\'import:', error);
            throw error;
        }
    }
}

// Fonction principale
async function main() {
    const importer = new CovidDataImporter();

    try {
        await importer.connect();
        await importer.importAll();
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

module.exports = CovidDataImporter;