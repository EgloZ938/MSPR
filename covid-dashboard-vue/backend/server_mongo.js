const express = require('express');
const { MongoClient } = require('mongodb');
const cors = require('cors');
const path = require('path');
const jwt = require('jsonwebtoken');
require('dotenv').config();

const app = express();
app.use(cors());
app.use(express.json());
app.use(express.static('public'));

// Configuration MongoDB
const MONGO_URI = process.env.MONGO_URI || 'mongodb+srv://theoevonwebmaster:<db_password>@cluster0.gpnsshs.mongodb.net/';
const DB_NAME = process.env.DB_NAME || 'covid_db';

let db;

// Connexion à MongoDB
async function connectMongoDB() {
    try {
        const client = new MongoClient(MONGO_URI);
        await client.connect();
        db = client.db(DB_NAME);
        console.log('✅ Connecté à MongoDB avec succès');
    } catch (error) {
        console.error('❌ Erreur de connexion MongoDB:', error);
        process.exit(1);
    }
}

// Middleware d'authentification (identique à PostgreSQL)
const authenticateToken = (req, res, next) => {
    const authHeader = req.headers['authorization'];
    const token = authHeader && authHeader.split(' ')[1];
    if (!token) {
        return res.status(401).json({ error: 'Token manquant' });
    }
    jwt.verify(token, process.env.JWT_SECRET || 'votre_secret_jwt', (err, user) => {
        if (err) {
            return res.status(403).json({ error: 'Token invalide' });
        }
        req.user = user;
        next();
    });
};

// Routes d'authentification (identiques à PostgreSQL)
app.post('/api/admin/login', (req, res) => {
    const { username, password } = req.body;
    if (username === process.env.ADMIN_USERNAME && password === process.env.ADMIN_PASSWORD) {
        const token = jwt.sign(
            { username: process.env.ADMIN_USERNAME },
            process.env.JWT_SECRET || 'votre_secret_jwt',
            { expiresIn: '1h' }
        );
        res.json({ token });
    } else {
        res.status(401).json({ error: 'Identifiants incorrects' });
    }
});

app.get('/api/admin/dashboard', authenticateToken, (req, res) => {
    res.json({ message: 'Accès autorisé au dashboard admin' });
});

// Route principale (identique à PostgreSQL)
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public/index.html'));
});

// API pour les statistiques globales (EXACTEMENT comme PostgreSQL)
app.get('/api/global-stats', async (req, res) => {
    try {
        // Équivalent: SELECT MAX(date) FROM daily_stats
        const latestDateResult = await db.collection('daily_stats')
            .aggregate([
                { $group: { _id: null, maxDate: { $max: "$date" } } }
            ]).toArray();

        if (latestDateResult.length === 0) {
            return res.json({ total_confirmed: 0, total_deaths: 0, total_recovered: 0, total_active: 0 });
        }

        const latestDate = latestDateResult[0].maxDate;

        // Équivalent exact de la requête PostgreSQL
        const pipeline = [
            { $match: { date: latestDate } },
            {
                $group: {
                    _id: null,
                    total_confirmed: { $sum: "$confirmed" },
                    total_deaths: { $sum: "$deaths" },
                    total_recovered: { $sum: "$recovered" },
                    total_active: { $sum: "$active" }
                }
            }
        ];

        const result = await db.collection('daily_stats').aggregate(pipeline).toArray();
        res.json(result[0] || { total_confirmed: 0, total_deaths: 0, total_recovered: 0, total_active: 0 });
    } catch (err) {
        console.error('Erreur global-stats:', err);
        res.status(500).json({ error: err.message });
    }
});

// API pour les top pays (EXACTEMENT comme PostgreSQL)
app.get('/api/top-countries', async (req, res) => {
    try {
        // Équivalent de la requête PostgreSQL avec JOIN
        const latestDateResult = await db.collection('daily_stats')
            .aggregate([{ $group: { _id: null, maxDate: { $max: "$date" } } }]).toArray();

        if (latestDateResult.length === 0) {
            return res.json([]);
        }

        const latestDate = latestDateResult[0].maxDate;

        const pipeline = [
            { $match: { date: latestDate } },
            {
                $lookup: {
                    from: "countries",
                    localField: "country_id",
                    foreignField: "_id",
                    as: "country"
                }
            },
            { $unwind: "$country" },
            {
                $addFields: {
                    mortality_rate: {
                        $cond: {
                            if: { $gt: ["$confirmed", 0] },
                            then: { $multiply: [{ $divide: ["$deaths", "$confirmed"] }, 100] },
                            else: 0
                        }
                    }
                }
            },
            { $sort: { confirmed: -1 } },
            { $limit: 10 },
            {
                $project: {
                    country_region: "$country.country_name",
                    confirmed: 1,
                    deaths: 1,
                    recovered: 1,
                    active: 1,
                    mortality_rate: 1
                }
            }
        ];

        const result = await db.collection('daily_stats').aggregate(pipeline).toArray();
        res.json(result);
    } catch (err) {
        console.error('Erreur top-countries:', err);
        res.status(500).json({ error: err.message });
    }
});

// API pour l'évolution mondiale (EXACTEMENT comme PostgreSQL)
app.get('/api/global-timeline', async (req, res) => {
    try {
        const pipeline = [
            {
                $group: {
                    _id: "$date",
                    confirmed: { $sum: "$confirmed" },
                    deaths: { $sum: "$deaths" },
                    recovered: { $sum: "$recovered" },
                    active: { $sum: "$active" }
                }
            },
            {
                $project: {
                    _id: 0,
                    date: "$_id",
                    confirmed: 1,
                    deaths: 1,
                    recovered: 1,
                    active: 1
                }
            },
            { $sort: { date: 1 } }
        ];

        const result = await db.collection('daily_stats').aggregate(pipeline).toArray();
        res.json(result);
    } catch (err) {
        console.error('Erreur global-timeline:', err);
        res.status(500).json({ error: err.message });
    }
});

// API pour les statistiques par région WHO (EXACTEMENT comme PostgreSQL)
app.get('/api/region-stats', async (req, res) => {
    try {
        const pipeline = [
            {
                $lookup: {
                    from: "countries",
                    localField: "country_id",
                    foreignField: "_id",
                    as: "country"
                }
            },
            { $unwind: "$country" },
            {
                $lookup: {
                    from: "who_regions",
                    localField: "country.who_region_id",
                    foreignField: "_id",
                    as: "who_region"
                }
            },
            { $unwind: "$who_region" },
            {
                $group: {
                    _id: {
                        region_name: "$who_region.region_name",
                        date: "$date"
                    },
                    confirmed: { $sum: "$confirmed" },
                    deaths: { $sum: "$deaths" },
                    recovered: { $sum: "$recovered" },
                    active: { $sum: "$active" }
                }
            },
            {
                $addFields: {
                    mortality_rate: {
                        $cond: {
                            if: { $gt: ["$confirmed", 0] },
                            then: { $multiply: [{ $divide: ["$deaths", "$confirmed"] }, 100] },
                            else: 0
                        }
                    }
                }
            },
            {
                $project: {
                    _id: 0,
                    region_name: "$_id.region_name",
                    date: "$_id.date",
                    confirmed: 1,
                    deaths: 1,
                    recovered: 1,
                    active: 1,
                    mortality_rate: 1
                }
            },
            { $sort: { date: 1 } }
        ];

        const result = await db.collection('daily_stats').aggregate(pipeline).toArray();
        res.json(result);
    } catch (err) {
        console.error('Erreur region-stats:', err);
        res.status(500).json({ error: err.message });
    }
});

// API pour l'évolution d'un pays spécifique (EXACTEMENT comme PostgreSQL)
app.get('/api/country-timeline/:country', async (req, res) => {
    try {
        const pipeline = [
            {
                $lookup: {
                    from: "countries",
                    localField: "country_id",
                    foreignField: "_id",
                    as: "country"
                }
            },
            { $unwind: "$country" },
            { $match: { "country.country_name": req.params.country } },
            {
                $addFields: {
                    mortality_rate: {
                        $cond: {
                            if: { $gt: ["$confirmed", 0] },
                            then: { $multiply: [{ $divide: ["$deaths", "$confirmed"] }, 100] },
                            else: 0
                        }
                    }
                }
            },
            { $sort: { date: 1 } },
            {
                $project: {
                    _id: 0,
                    date: 1,
                    confirmed: 1,
                    deaths: 1,
                    recovered: 1,
                    active: 1,
                    mortality_rate: 1
                }
            }
        ];

        const result = await db.collection('daily_stats').aggregate(pipeline).toArray();
        res.json(result);
    } catch (err) {
        console.error('Erreur country-timeline:', err);
        res.status(500).json({ error: err.message });
    }
});

// API pour la comparaison de taux entre pays (EXACTEMENT comme PostgreSQL)
app.get('/api/country-comparison', async (req, res) => {
    const countries = req.query.countries ? req.query.countries.split(',') : [];
    if (countries.length === 0) {
        return res.status(400).json({ error: 'Aucun pays spécifié' });
    }

    try {
        const pipeline = [
            {
                $lookup: {
                    from: "countries",
                    localField: "country_id",
                    foreignField: "_id",
                    as: "country"
                }
            },
            { $unwind: "$country" },
            { $match: { "country.country_name": { $in: countries } } },
            {
                $addFields: {
                    mortality_rate: {
                        $cond: {
                            if: { $gt: ["$confirmed", 0] },
                            then: { $multiply: [{ $divide: ["$deaths", "$confirmed"] }, 100] },
                            else: 0
                        }
                    }
                }
            },
            { $sort: { date: 1 } },
            {
                $project: {
                    _id: 0,
                    country_name: "$country.country_name",
                    date: 1,
                    confirmed: 1,
                    deaths: 1,
                    recovered: 1,
                    active: 1,
                    mortality_rate: 1
                }
            }
        ];

        const result = await db.collection('daily_stats').aggregate(pipeline).toArray();
        res.json(result);
    } catch (err) {
        console.error('Erreur country-comparison:', err);
        res.status(500).json({ error: err.message });
    }
});

// API pour les données filtrées (EXACTEMENT comme PostgreSQL)
app.get('/api/filtered-data', async (req, res) => {
    const { region, dateStart, dateEnd, minCases, maxCases } = req.query;

    let matchConditions = {};

    if (dateStart) {
        matchConditions.date = { $gte: new Date(dateStart) };
    }
    if (dateEnd) {
        if (matchConditions.date) {
            matchConditions.date.$lte = new Date(dateEnd);
        } else {
            matchConditions.date = { $lte: new Date(dateEnd) };
        }
    }
    if (minCases) {
        matchConditions.confirmed = { $gte: parseInt(minCases) };
    }
    if (maxCases) {
        if (matchConditions.confirmed) {
            matchConditions.confirmed.$lte = parseInt(maxCases);
        } else {
            matchConditions.confirmed = { $lte: parseInt(maxCases) };
        }
    }

    try {
        const pipeline = [
            { $match: matchConditions },
            {
                $lookup: {
                    from: "countries",
                    localField: "country_id",
                    foreignField: "_id",
                    as: "country"
                }
            },
            { $unwind: "$country" },
            {
                $lookup: {
                    from: "who_regions",
                    localField: "country.who_region_id",
                    foreignField: "_id",
                    as: "who_region"
                }
            },
            { $unwind: "$who_region" }
        ];

        if (region) {
            pipeline.push({ $match: { "who_region.region_name": region } });
        }

        pipeline.push(
            {
                $addFields: {
                    mortality_rate: {
                        $cond: {
                            if: { $gt: ["$confirmed", 0] },
                            then: { $multiply: [{ $divide: ["$deaths", "$confirmed"] }, 100] },
                            else: 0
                        }
                    }
                }
            },
            { $sort: { confirmed: -1 } },
            {
                $project: {
                    _id: 0,
                    country_name: "$country.country_name",
                    region_name: "$who_region.region_name",
                    date: 1,
                    confirmed: 1,
                    deaths: 1,
                    recovered: 1,
                    active: 1,
                    mortality_rate: 1
                }
            }
        );

        const result = await db.collection('daily_stats').aggregate(pipeline).toArray();
        res.json(result);
    } catch (err) {
        console.error('Erreur filtered-data:', err);
        res.status(500).json({ error: err.message });
    }
});

// API pour obtenir la liste de tous les pays (EXACTEMENT comme PostgreSQL)
app.get('/api/countries', async (req, res) => {
    try {
        const result = await db.collection('countries')
            .find({})
            .project({
                id: "$_id",
                country_name: 1,
                continent: 1,
                population: 1
            })
            .sort({ country_name: 1 })
            .toArray();

        res.json(result);
    } catch (err) {
        console.error('Erreur countries:', err);
        res.status(500).json({ error: err.message });
    }
});

// API pour obtenir les détails actuels d'un pays spécifique (EXACTEMENT comme PostgreSQL)
app.get('/api/country-details/:country', async (req, res) => {
    try {
        const pipeline = [
            {
                $lookup: {
                    from: "countries",
                    localField: "country_id",
                    foreignField: "_id",
                    as: "country"
                }
            },
            { $unwind: "$country" },
            { $match: { "country.country_name": req.params.country } },
            { $sort: { last_updated: -1 } },
            { $limit: 1 },
            {
                $project: {
                    _id: 0,
                    country_name: "$country.country_name",
                    population: "$country.population",
                    total_tests: 1,
                    tests_per_million: 1,
                    cases_per_million: 1,
                    deaths_per_million: 1,
                    serious_critical: 1,
                    last_updated: 1
                }
            }
        ];

        const result = await db.collection('country_details').aggregate(pipeline).toArray();

        if (result.length > 0) {
            res.json(result[0]);
        } else {
            res.status(404).json({ error: 'Pays non trouvé ou aucune donnée disponible' });
        }
    } catch (err) {
        console.error('Erreur country-details:', err);
        res.status(500).json({ error: err.message });
    }
});

// Initialisation et démarrage du serveur
async function startServer() {
    await connectMongoDB();

    const PORT = process.env.PORT || 3000;
    app.listen(PORT, () => {
        console.log(`🚀 Serveur COVID Dashboard démarré sur http://localhost:${PORT}`);
        console.log(`📊 Base de données: ${DB_NAME}`);
        console.log(`🔗 MongoDB Atlas connecté`);
    });
}

startServer().catch(console.error);