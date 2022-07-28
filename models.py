from sqlalchemy import *
from sqlalchemy.types import JSON
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from functools import partial

Base = declarative_base()
Nullable = partial(Column, nullable=True)

_engine = None
_SessionMaker = None
def get_db():
    global _engine, _SessionMaker
    _engine = create_engine('sqlite:///database.sqlite', echo=False, future=True)
    _SessionMaker = sessionmaker(_engine)
    return _engine

def get_session():
    return _SessionMaker()
def with_session(f):
    def f2(*args, **kwargs):
        with _SessionMaker() as session:
            with session.begin():
                return f(session, *args, **kwargs)
    return f2


class Avatar(Base):
    __tablename__ = "avatars"

    id = Column(String, primary_key=True)
    comment = Nullable(String)
    dataset = Column(String)
    extra = Nullable(JSON)

    model_id = Nullable(Integer, ForeignKey("models.id"))
    model = relationship("Model", back_populates="avatars")

    def __repr__(self):
        # TODO: use dir to auto generate?
        return f"Avatar(id={self.id!r})"


class Model(Base):
    __tablename__ = "models"

    id = Column(Integer, primary_key=True)
    name = Nullable(String, index=True)
    creator = Nullable(String)
    category = Nullable(String)
    url = Nullable(String)
    comment = Nullable(String)
    extra = Nullable(JSON)

    avatars = relationship("Avatar", back_populates="model")

    def __repr__(self):
        return f"Model(id={self.id!r})"


# TODO: model name<>alias table? with full text search?
# TODO: avatar<>images table?
# TODO: avatar<>credit (who sauce it, discord annotater etc?)
